import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .ss_trainer_ETP import ETPTrainer


@baseline_registry.register_trainer(name="SS-VIL-ETP")
class ViewInvariantETPTrainer(ETPTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len)
                
    def reconfig(self, position_shift=[0.0, 0.0, 0.0], orientation_shift=[0.0, 0.0, 0.0]):
        
        self.config.defrost()
        depth_sensors = [
                'DEPTH_SENSOR', 'DEPTH_30', 'DEPTH_60', 'DEPTH_90', 'DEPTH_120',
                'DEPTH_150', 'DEPTH_180', 'DEPTH_210', 'DEPTH_240',
                'DEPTH_270', 'DEPTH_300', 'DEPTH_330'
            ]
        rgb_sensors = [item.replace('DEPTH','RGB') for item in depth_sensors]
        sensors = rgb_sensors + depth_sensors
        for sensor in sensors:
            position = self.config.TASK_CONFIG.SIMULATOR[sensor].POSITION
            self.config.TASK_CONFIG.SIMULATOR[sensor+'_AUG'].POSITION = [a+b for a, b in zip(position, position_shift)]
            orientation = self.config.TASK_CONFIG.SIMULATOR[sensor].ORIENTATION
            self.config.TASK_CONFIG.SIMULATOR[sensor+'_AUG'].ORIENTATION = [a+b for a, b in zip(orientation, orientation_shift)] 
        print('std cfg position: ', self.config.TASK_CONFIG.SIMULATOR['RGB_SENSOR'].POSITION, self.config.TASK_CONFIG.SIMULATOR['RGB_SENSOR'].ORIENTATION)
        print('new aug cfg position: ', self.config.TASK_CONFIG.SIMULATOR['RGB_SENSOR_AUG'].POSITION, self.config.TASK_CONFIG.SIMULATOR['RGB_SENSOR_AUG'].ORIENTATION)
        self.config.freeze()
        
    def _set_config_aug(self):
        self.config.defrost()
        rgb_sensors = [item for item in self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS if 'RGB' in item]
        assert len(rgb_sensors) == 12, "number of rgb sensors should be 12"
        for rgb_sensor in rgb_sensors:
            depth_sensor = rgb_sensor.replace('RGB', 'DEPTH')
            self.config.TASK_CONFIG.SIMULATOR[rgb_sensor+'_AUG'] = deepcopy(self.config.TASK_CONFIG.SIMULATOR[rgb_sensor])
            uuid = 'rgb' if not hasattr(self.config.TASK_CONFIG.SIMULATOR[rgb_sensor], 'UUID') else self.config.TASK_CONFIG.SIMULATOR[rgb_sensor].UUID 
            self.config.TASK_CONFIG.SIMULATOR[rgb_sensor+'_AUG'].UUID = uuid + '_aug'
            self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(rgb_sensor+'_AUG')
            self.config.TASK_CONFIG.SIMULATOR[depth_sensor+'_AUG'] = deepcopy(self.config.TASK_CONFIG.SIMULATOR[depth_sensor])
            self.config.TASK_CONFIG.SIMULATOR[depth_sensor+'_AUG'].UUID = uuid.replace('rgb', 'depth') + '_aug'
            self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(depth_sensor+'_AUG')
        self.config.freeze()
    
    
    def train(self):
        self._set_config()
        self._set_config_aug()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        logger.info('Traning Starts... GOOD LUCK!')
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval
            
            
            def prepare_config_envs(index=0):
                if hasattr(self, 'envs') and self.envs is not None:
                    self.envs.close()
                
                if self.config.IL.view_sampling == 'height_only':
                    np.random.seed(int(os.environ['LOCAL_RANK'])+index)
                    print('random seed: ', int(os.environ['LOCAL_RANK'])+index)
                    random_values = np.random.random()
                    max_height_shift = self.config.IL.max_height_shift
                    height_shift = (random_values - 0.5) * max_height_shift
                    self.reconfig(position_shift=[0, height_shift, 0])
                elif self.config.IL.view_sampling == 'height_angle':
                    np.random.seed(int(os.environ['LOCAL_RANK'])+index)
                    print('random seed: ', int(os.environ['LOCAL_RANK'])+index)
                    random_values = np.random.random()
                    max_height_shift = self.config.IL.max_height_shift
                    height_shift = (random_values - 0.5) / 0.5 * max_height_shift
                    max_angle_shift = self.config.IL.max_angle_shift
                    max_angle = - max_angle_shift * height_shift / max_height_shift # height_shift < 0 -> looking up
                    angle = np.random.uniform(min(0, max_angle), max(0, max_angle))
                    euler_angle = angle * np.pi / 180
                    self.reconfig(position_shift=[0, height_shift, 0], orientation_shift=[euler_angle, 0, 0])
                    
                
                if self.config.MODEL.task_type == 'rxr':
                    self.gt_data = {}
                    for role in self.config.TASK_CONFIG.DATASET.ROLES:
                        with gzip.open(
                            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                                split=self.split, role=role
                            ), "rt") as f:
                            self.gt_data.update(json.load(f))
                self._init_envs()
        
            
            if idx % (self.config.IL.resample_interval // self.config.IL.log_every) == 0:
                prepare_config_envs(index=cur_iter)

            sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
            # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
            logs = self._train_interval(interval, self.config.IL.ml_weight, self.config.IL.cl_weight, self.config.IL.wp_cl_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)

    def _train_interval(self, interval, ml_weight, cl_weight, wp_cl_weight, sample_ratio):
        # self.policy.train()
        # if self.world_size > 1:
        #     self.policy.net.module.rgb_encoder.eval()
        #     self.policy.net.module.depth_encoder.eval()
        # else:
        #     self.policy.net.rgb_encoder.eval()
        #     self.policy.net.depth_encoder.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            self.optimizer.zero_grad()
            self.loss = 0.

            with autocast():
                self.rollout('train', ml_weight, cl_weight, wp_cl_weight, sample_ratio)
            self.scaler.scale(self.loss).backward() # self.loss.backward()
            self.scaler.step(self.optimizer)        # self.optimizer.step()
            self.scaler.update()

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
            
        return deepcopy(self.logs)

    def rollout(self, mode, ml_weight=None, cl_weight=None, wp_cl_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError
        
        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )
        
        loss = 0.
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config.IL.loc_noise, 
                               self.config.MODEL.merge_ghost,
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs

        for stepk in range(self.max_len):
            
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]
            
            if mode == 'train':
                cl_loss = self.policy.net(
                    mode = "contrastive",
                    observations = batch
                )
                
                wp_cl_loss = self.policy.net(
                    mode = "waypoint_contrastive",
                    observations = batch,
                    in_train = (mode == 'train' and self.config.IL.waypoint_aug),
                )
                
            
            # cand waypoint prediction
            wp_outputs = self.policy.net(
                mode = "waypoint",
                observations = batch,
                in_train = (mode == 'train' and self.config.IL.waypoint_aug),
            )
            

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)
            
            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1]
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                           cur_vp[i], cur_pos[i], cur_embeds,
                                           cand_vp[i], cand_pos[i], cand_embeds,
                                           cand_real_pos[i])

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            # random sample demo
            # logits = torch.randn(nav_inputs['gmap_masks'].shape).cuda()
            # logits.masked_fill_(~nav_inputs['gmap_masks'], -float('inf'))
            # logits.masked_fill_(nav_inputs['gmap_visited_masks'], -float('inf'))

            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)

            # determine action
            if feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float)<=sample_ratio, teacher_actions, a_t)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.logs['IL_loss'].append(loss.item())
            if cl_loss is None or wp_cl_loss is None:
                self.loss += loss
            else:
                self.logs['raw CL_loss'].append(cl_loss.item())
                cl_loss = cl_weight * cl_loss
                self.logs['CL_loss'].append(cl_loss.item())
                self.logs['raw WP_CL_loss'].append(wp_cl_loss.item())
                wp_cl_loss = wp_cl_weight * wp_cl_loss
                self.logs['WP_CL_loss'].append(wp_cl_loss.item())
                self.loss += loss + cl_loss + wp_cl_loss
       
    
     