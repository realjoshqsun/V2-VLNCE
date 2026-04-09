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

@baseline_registry.register_trainer(name="eval-vv-grid-ETP")
class ETPTrainer(ETPTrainer):

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

        if self.config.EVAL.distribution == 'equal_spacing':
            # Generate Uniform Grid with Equal Spacing
            num_grid = self.config.IL.num_grid  # Controls grid density, NUM_POINTS = num_grid ** 2 - 1
            max_height_shift = self.config.IL.max_height_shift # meters
            max_angle_shift = self.config.IL.max_angle_shift # degrees

            height_shifts = np.linspace(0, max_height_shift, num_grid)
            angle_shifts = np.linspace(0, max_angle_shift, num_grid)

            H, A = np.meshgrid(height_shifts, angle_shifts)
            valid_mask = (A / max_angle_shift <= H / max_height_shift)
            height_shifts = H[valid_mask].flatten()
            angle_shifts = A[valid_mask].flatten()

            height_shifts = np.concatenate([height_shifts, -height_shifts])
            angle_shifts = - np.concatenate([angle_shifts, -angle_shifts])
            height_angle_shifts = np.column_stack((H.flatten(), A.flatten()))
        
        elif self.config.EVAL.distribution == 'square':
            num_grid = self.config.IL.num_grid  # Controls grid density, NUM_POINTS = num_grid ** 2
            max_height_shift = self.config.IL.max_height_shift # meters
            max_angle_shift = self.config.IL.max_angle_shift # degrees

            height_shifts = np.linspace(-max_height_shift, max_height_shift, num_grid)
            angle_shifts = np.linspace(-max_angle_shift, max_angle_shift, num_grid)
            H, A = np.meshgrid(height_shifts, angle_shifts)
            height_angle_shifts = np.column_stack((H.flatten(), A.flatten()))
        
        elif self.config.EVAL.distribution == 'custom':
            # Read from config
            shift_list = self.config.EVAL.custom_height_angle_shifts
            height_angle_shifts = np.array(shift_list)
        
        print('height_angle_shifts:', height_angle_shifts)
        
        # Save Base Cfg
        self.config.defrost()
        depth_sensors = [
                'DEPTH_SENSOR', 'DEPTH_30', 'DEPTH_60', 'DEPTH_90', 'DEPTH_120',
                'DEPTH_150', 'DEPTH_180', 'DEPTH_210', 'DEPTH_240',
                'DEPTH_270', 'DEPTH_300', 'DEPTH_330'
            ]
        rgb_sensors = [item.replace('DEPTH','RGB') for item in depth_sensors]
        sensors = rgb_sensors + depth_sensors
        sensor_base_cfg = {}
        for sensor in sensors:
            position = self.config.TASK_CONFIG.SIMULATOR[sensor].POSITION
            orientation = self.config.TASK_CONFIG.SIMULATOR[sensor].ORIENTATION
            sensor_base_cfg[sensor] = {'POSITION': position, 'ORIENTATION': orientation}
        self.config.freeze()
        
        for i, (height_shift, angle_shift) in enumerate(height_angle_shifts):
            print(i, height_shift, angle_shift)
            os.makedirs(
                os.path.join(
                    self.config.RESULTS_DIR,
                    f"Grid_{i}_height-shift={height_shift}_angle-shift={angle_shift}/"
                    ), 
                exist_ok=True)
            
            # Add Grid Height-Angle Shift to Base Cfg
            self.config.defrost()
            euler_angle = angle_shift * np.pi / 180
            for sensor in sensors:
                self.config.TASK_CONFIG.SIMULATOR[sensor].POSITION = [a+b for a, b in zip(sensor_base_cfg[sensor]['POSITION'], [0, height_shift, 0])] 
                self.config.TASK_CONFIG.SIMULATOR[sensor].ORIENTATION = [a+b for a, b in zip(sensor_base_cfg[sensor]['ORIENTATION'], [euler_angle, 0, 0])] 
            self.config.freeze()

            
            # Evaluation Started for Grid-i
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"Grid_{i}_height-shift={height_shift}_angle-shift={angle_shift}/",
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
            self.waypoint_predictor.eval()

            if self.config.EVAL.EPISODE_COUNT == -1:
                eps_to_eval = sum(self.envs.number_of_episodes)
            else:
                eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
            self.stat_eps = {}
            self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

            while len(self.stat_eps) < eps_to_eval:
                self.rollout('eval')
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
                f"Grid_{i}_height-shift={height_shift}_angle-shift={angle_shift}/",
                f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
            )
            with open(fname, "w") as f:
                json.dump(self.stat_eps, f, indent=2)

            if self.local_rank < 1:
                if self.config.EVAL.SAVE_RESULTS:
                    fname = os.path.join(
                        self.config.RESULTS_DIR,
                        f"Grid_{i}_height-shift={height_shift}_angle-shift={angle_shift}/",
                        f"stats_ckpt_{checkpoint_index}_{split}.json",
                    )
                    with open(fname, "w") as f:
                        json.dump(aggregated_states, f, indent=2)

                logger.info(f"Episodes evaluated: {total}")
                checkpoint_num = checkpoint_index + 1
                for k, v in aggregated_states.items():
                    logger.info(f"Average episode {k}: {v:.6f}")
       
