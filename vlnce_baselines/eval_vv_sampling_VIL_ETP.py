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

from .ss_trainer_VIL_ETP_final import VIL_ETPTrainer as ETPTrainer


@baseline_registry.register_trainer(name="eval-vv-sampling-VIL-ETP")
class VIL_ETPTrainer(ETPTrainer):
    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            eval=True,
        )
        self.policy.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None


        # Please ensure using 1 GPU and NUM_ENV is set to be 1 !
        sensor_base_cfg = {}
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
            orientation = self.config.TASK_CONFIG.SIMULATOR[sensor].ORIENTATION
            sensor_base_cfg[sensor] = {'POSITION': position, 'ORIENTATION': orientation}
        self.config.freeze()
        print(sensor_base_cfg)
        np.random.seed(128)
        
        assert self.config.GPU_NUMBERS == 1 and self.config.NUM_ENVIRONMENTS == 1
        while len(self.stat_eps) < eps_to_eval:
            
            self.envs.close()
            if self.config.IL.view_sampling == 'height_only':
                max_height_shift = self.config.IL.max_height_shift
                height_shift = np.random.uniform(-max_height_shift, max_height_shift)
            elif self.config.IL.view_sampling == 'height_angle':

                # max_height_shift = self.config.IL.max_height_shift
                # height_shift = np.random.uniform(-max_height_shift, max_height_shift)
                # max_angle_shift = self.config.IL.max_angle_shift
                # max_angle = - max_angle_shift * height_shift / max_height_shift # height_shift < 0 -> looking up
                # angle = np.random.uniform(min(0, max_angle), max(0, max_angle))
                # euler_angle = angle * np.pi / 180
                
                # Sample uniformly within a right triangle
                u = np.sqrt(np.random.uniform(0, 1))  # sqrt ensures uniformity in triangle
                v = np.random.uniform(0, u)  # Ensures points are inside the triangle

                max_height_shift = self.config.IL.max_height_shift
                max_angle_shift = self.config.IL.max_angle_shift

                # Randomly assign positive or negative sign, ensuring height_shift and angle_shift match
                sign = np.random.choice([-1, 1])
                height_shift = sign * u * max_height_shift
                angle_shift = sign * v * max_angle_shift
                euler_angle = angle_shift * np.pi / 180
                
            elif self.config.IL.view_sampling == 'height_angle_2d_uniform':
                height_shift = np.random.uniform(-self.config.IL.max_height_shift, self.config.IL.max_height_shift)
                angle_shift = np.random.uniform(-self.config.IL.max_angle_shift, self.config.IL.max_angle_shift)
                euler_angle = angle_shift * np.pi / 180
                
            elif self.config.IL.view_sampling == 'height_angle_2d_uniform_with_range':
                height_shift = np.random.uniform(self.config.IL.min_height, self.config.IL.max_height)
                angle_shift = np.random.uniform(self.config.IL.min_angle, self.config.IL.max_angle)
                euler_angle = angle_shift * np.pi / 180
                
            self.config.defrost()
            print('sampled height shift:', height_shift)
            if euler_angle is not None:
                print('sampled angle shift:', euler_angle)
            for sensor in sensors:
                self.config.TASK_CONFIG.SIMULATOR[sensor].POSITION = [a+b for a, b in zip(sensor_base_cfg[sensor]['POSITION'], [0, height_shift, 0])] 
                if euler_angle is not None:
                    self.config.TASK_CONFIG.SIMULATOR[sensor].ORIENTATION = [a+b for a, b in zip(sensor_base_cfg[sensor]['ORIENTATION'], [euler_angle, 0, 0])] 
            self.config.freeze()
            # if self.config.MODEL.task_type == 'rxr':
            #     self.gt_data = {}
            #     for role in self.config.TASK_CONFIG.DATASET.ROLES:
            #         with gzip.open(
            #             self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
            #                 split=self.split, role=role
            #             ), "rt") as f:
            #             self.gt_data.update(json.load(f))
            
            self.envs = construct_envs(
                self.config, 
                get_env_class(self.config.ENV_NAME),
                episodes_allowed=self.traj[len(self.stat_eps):len(self.stat_eps)+1:],
                auto_reset_done=False, # unseen: 11006 
            )
            
            self.rollout('eval')
            print('Done / All:', len(self.stat_eps), eps_to_eval)
        self.envs.close()
        

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            eval=True,
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}   # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)


        # Please ensure using 1 GPU and NUM_ENV is set to be 1 !
        sensor_base_cfg = {}
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
            orientation = self.config.TASK_CONFIG.SIMULATOR[sensor].ORIENTATION
            sensor_base_cfg[sensor] = {'POSITION': position, 'ORIENTATION': orientation}
        self.config.freeze()
        print(sensor_base_cfg)
        np.random.seed(42)
        
        assert self.config.GPU_NUMBERS == 1 and self.config.NUM_ENVIRONMENTS == 1
        for i in range(44):
            print(i)
        self.envs.close()

