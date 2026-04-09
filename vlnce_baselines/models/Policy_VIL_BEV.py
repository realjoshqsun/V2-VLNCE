from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.models.bev.vlnbert_init import get_vlnbert_models
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from vlnce_baselines.models.encoders.resnet_encoders_bev import CLIPEncoderB16
from vlnce_baselines.models.policy import ILPolicy

from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, angle_feature_torch, length2mask)
import math

@baseline_registry.register_policy
class PolicyVIL_BEV(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            VILBEV(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )

class Critic(nn.Module):
    def __init__(self, drop_ratio):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class VILBEV(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions,
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nLoading Waypoint Predictor to BEV model ...')
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor_teacher = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if model_config.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor_teacher.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor_teacher.parameters():
            param.requires_grad_(False)
        self.waypoint_predictor_teacher.eval()
            
        self.waypoint_predictor_student = BinaryDistPredictor_TRM(device=self.device)
        self.waypoint_predictor_student.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor_student.parameters():
            param.requires_grad_(False)
        self.waypoint_predictor_student.eval()
        
        for param in self.waypoint_predictor_student.visual_fc_depth.parameters():
            param.requires_grad_(True)
        self.waypoint_predictor_student.visual_fc_depth.train()

        print('\nInitalizing the BEV model ...')
        
        self.temperature = getattr(model_config, "temperature", 1)
        self.keep_origin_view_prob = 1
        self.keep_origin_view_wp_prob = 1
                    
        self.rgb_prediction = nn.Linear(512, 512, bias=False)
        self.rgb_prediction.weight.data = torch.eye(512, 512)
        self.depth_prediction = nn.Linear(128, 128, bias=False)
        self.depth_prediction.weight.data = torch.eye(128, 128)
        self.rgb_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.depth_projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
            
        self.keep_origin_view_prob = getattr(model_config, "keep_origin_view_prob", 1)
        self.keep_origin_view_wp_prob = getattr(model_config, "keep_origin_view_wp_prob", 1)

        print('\nInitalizing the BEV model ...')
        self.temperature = getattr(model_config, "temperature", 1)
        self.keep_origin_view_prob = 1
        self.keep_origin_view_wp_prob = 1
                    
        self.rgb_prediction = nn.Linear(512, 512, bias=False)
        self.rgb_prediction.weight.data = torch.eye(512, 512)
        self.depth_prediction = nn.Linear(128, 128, bias=False)
        self.depth_prediction.weight.data = torch.eye(128, 128)
        self.rgb_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.depth_projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
            
        self.keep_origin_view_prob = getattr(model_config, "keep_origin_view_prob", 1)
        self.keep_origin_view_wp_prob = getattr(model_config, "keep_origin_view_wp_prob", 1)
        
        self.vln_bert = get_vlnbert_models(config=model_config)
        self.drop_env = nn.Dropout(p=0.4)

        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )
        self.space_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
        self.grid_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((14,14)), nn.Flatten(start_dim=1))

        self.rgb_encoder = CLIPEncoderB16(self.device)
        self.space_pool_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
    
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)                          # counter-clockwise
        pano_angle_rad_c = (1-self.pano_img_idxes/12) * 2 * math.pi                     # clockwise  array([360, 330, 300, 270, 240, 210, 180, 150, 120,  90,  60, 30])
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))   # clockwise

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(self, mode=None, 
                txt_ids=None, txt_masks=None, txt_embeds=None, 
                waypoint_predictor=None, observations=None, in_train=True,
                rgb_fts=None, dep_fts=None, loc_fts=None, 
                nav_types=None, view_lens=None,
                gmap_vp_ids=None, gmap_step_ids=None,
                gmap_img_fts=None, gmap_pos_fts=None, 
                gmap_masks=None, gmap_visited_masks=None, gmap_pair_dists=None,
                bev_fts=None, bev_pos_fts=None,
                bev_masks=None, bev_nav_masks=None, 
                bev_cand_idxs=None, bev_cand_vpids=None,
                waypoint_heatmap_logits_teacher=None, waypoint_heatmap_logits_student=None):

        if mode == 'language':
            encoded_sentence = self.vln_bert.forward_txt(
                txt_ids, txt_masks,
            )
            return encoded_sentence

        elif mode == "contrastive":
            
            NUM_IMGS = 12
            def forward_obs(obs, suffix=''):
                batch_size = observations['rgb'].shape[0]
                depth_sensors = [
                    'depth', 'depth_30', 'depth_60', 'depth_90', 'depth_120',
                    'depth_150', 'depth_180', 'depth_210', 'depth_240',
                    'depth_270', 'depth_300', 'depth_330'
                ]
                rgb_sensors = [item.replace('depth', 'rgb') for item in depth_sensors]
                sensors = rgb_sensors + depth_sensors
                sensors = [item + suffix for item in sensors]
                
                depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
                rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)
                a_count = 0
                for i, k in enumerate(sensors):
                    if 'depth' in k: 
                        v = observations.get(k)
                        for bi in range(v.size(0)):
                            ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                            depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                            rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                        a_count += 1
                obs_view12 = {}
                obs_view12['depth'] = depth_batch
                obs_view12['rgb'] = rgb_batch
                depth_embeds = self.depth_encoder(obs_view12)     
                depth_grid_embeds = self.grid_pool_depth(depth_batch.permute(0, 3, 1, 2))
                rgb_embeds, rgb_grid_embeds = self.rgb_encoder(obs_view12)  
                
                # reverse the order of images back to counter-clockwise
                rgb_embeds_reshape = rgb_embeds.reshape(batch_size, NUM_IMGS, 512, 1, 1)
                rgb_feats = torch.cat(
                    [rgb_embeds_reshape[:,0:1,:], torch.flip(rgb_embeds_reshape[:,1:,:], [1])], dim=1
                )

                depth_embeds_reshape = depth_embeds.reshape(batch_size, NUM_IMGS, 128, 4, 4)
                depth_feats = torch.cat(
                    [depth_embeds_reshape[:,0:1,:], torch.flip(depth_embeds_reshape[:,1:,:], [1])], dim=1
                )

                rgb_feats = self.space_pool_rgb(rgb_feats)
                depth_feats = self.space_pool_depth(depth_feats)
                
                return rgb_feats, depth_feats
                
            rgb_feats, depth_feats = forward_obs(observations)
            rgb_feats_aug, depth_feats_aug = forward_obs(observations, suffix='_aug')
            
            bs, num_views, _ = rgb_feats.shape
            
            proj_original = torch.cat([self.rgb_projection(self.rgb_prediction(rgb_feats.reshape(bs*num_views, -1))).reshape(bs, num_views, -1), self.depth_projection(self.depth_prediction(depth_feats.reshape(bs*num_views, -1))).reshape(bs, num_views, -1)], dim=-1)
            proj_aug = torch.cat([self.rgb_projection(self.rgb_prediction(rgb_feats_aug.reshape(bs*num_views, -1))).reshape(bs, num_views, -1), self.depth_projection(self.depth_prediction(depth_feats_aug.reshape(bs*num_views, -1))).reshape(bs, num_views, -1)], dim=-1)
            
            original_views = F.normalize(proj_original, dim=-1)
            augmented_views = F.normalize(proj_aug, dim=-1)
            
            neg1 = torch.roll(original_views, shifts=num_views//2, dims=1)  # [bs, NUM_VIEWS, feat_dim]
            neg2 = torch.roll(original_views, shifts=1, dims=0)  # [bs, NUM_VIEWS, feat_dim]
            
            anchors = original_views.reshape(bs*num_views, -1)
            positives = augmented_views.reshape(bs*num_views, -1)
            neg1 = neg1.reshape(bs*num_views, -1)
            neg2 = neg2.reshape(bs*num_views, -1)
            
            logits_pos = torch.sum(anchors * positives, dim=-1) / self.temperature
            logits_neg1 = torch.sum(anchors * neg1, dim=-1) / self.temperature
            logits_neg2 = torch.sum(anchors * neg2, dim=-1) / self.temperature
            
            logits = torch.stack([logits_pos, logits_neg1, logits_neg2], dim=1)  # [bs*NUM_VIEWS, 3]
            labels = torch.zeros(bs * num_views, device=logits.device, dtype=torch.long)
            
            loss = F.cross_entropy(logits, labels)
            
            return loss
        
        
        elif mode == 'waypoint_distillation':
            batch_size = observations['rgb'].shape[0]
            NUM_IMGS = 12
            
            def forward_obs(observations, suffix=''):
                depth_sensors = [
                    'depth', 'depth_30', 'depth_60', 'depth_90', 'depth_120',
                    'depth_150', 'depth_180', 'depth_210', 'depth_240',
                    'depth_270', 'depth_300', 'depth_330'
                ]
                rgb_sensors = [item.replace('depth', 'rgb') for item in depth_sensors]
                sensors = rgb_sensors + depth_sensors
                sensors = [item + suffix for item in sensors]
                
                depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
                rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)
                a_count = 0
                for _, k in enumerate(sensors):
                    if 'depth' in k: 
                        v = observations.get(k)
                        for bi in range(v.size(0)):
                            ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                            depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                            rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                        a_count += 1
                obs_view12 = {}
                obs_view12['depth'] = depth_batch
                obs_view12['rgb'] = rgb_batch
                depth_embeds = self.depth_encoder(obs_view12)     
                depth_grid_embeds = self.grid_pool_depth(depth_batch.permute(0, 3, 1, 2))
                rgb_embeds, rgb_grid_embeds = self.rgb_encoder(obs_view12)  
                return rgb_embeds, rgb_grid_embeds, depth_embeds, depth_grid_embeds
                
            rgb_embedding, rgb_grid_embeds, depth_embedding, depth_grid_embeds = forward_obs(observations)
            rgb_embedding_aug, rgb_grid_embeds, depth_embedding_aug, depth_grid_embeds = forward_obs(observations, suffix='_aug')
            
            waypoint_heatmap_logits_teacher = self.waypoint_predictor_teacher(
                rgb_embedding, depth_embedding)
            
            waypoint_heatmap_logits_student = self.waypoint_predictor_student(
                rgb_embedding_aug, depth_embedding_aug)
            
            NUM_ANGLES = 120    # 120 angles 3 degrees each
            NUM_CLASSES = 12    # 12 distances at each sector
            waypoint_heatmap_logits_teacher_softmax = torch.softmax(
               waypoint_heatmap_logits_teacher.reshape(
                    batch_size, NUM_ANGLES*NUM_CLASSES,
                ), dim=1
            )
            waypoint_heatmap_logits_student_logsoftmax = torch.log_softmax(
               waypoint_heatmap_logits_student.reshape(
                    batch_size, NUM_ANGLES*NUM_CLASSES,
                ), dim=1
            )
            
            
            waypoint_distill_loss = F.kl_div(waypoint_heatmap_logits_student_logsoftmax, 
                                             waypoint_heatmap_logits_teacher_softmax, reduction='batchmean')
            return waypoint_distill_loss, waypoint_heatmap_logits_teacher, waypoint_heatmap_logits_student
        
        elif mode == 'waypoint':
            batch_size = observations['rgb'].shape[0]
            depth_sensors = [
                'depth', 'depth_30', 'depth_60', 'depth_90', 'depth_120',
                'depth_150', 'depth_180', 'depth_210', 'depth_240',
                'depth_270', 'depth_300', 'depth_330'
            ]
            rgb_sensors = [item.replace('depth', 'rgb') for item in depth_sensors]
            sensors = rgb_sensors + depth_sensors
            
            if self.keep_origin_view_prob < 1:
                if np.random.rand() < self.keep_origin_view_prob:
                    sensors = [item + '_aug' for item in sensors]
            
            
            ''' encoding rgb/depth at all directions ----------------------------- '''
            NUM_ANGLES = 120    # 120 angles 3 degrees each
            NUM_IMGS = 12
            NUM_CLASSES = 12    # 12 distances at each sector
            depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)
            # reverse the order of input images to clockwise
            # because waypoint predictor takes clockwise inputs
            a_count = 0
            for i, k in enumerate(sensors):
                if 'depth' in k:  # You might need to double check the keys order
                    v = observations.get(k)
                    for bi in range(v.size(0)):
                        ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                        depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                        rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                    a_count += 1
            obs_view12 = {}
            obs_view12['depth'] = depth_batch
            obs_view12['rgb'] = rgb_batch
            depth_embeds = self.depth_encoder(obs_view12)     
            depth_grid_embeds = self.grid_pool_depth(depth_batch.permute(0, 3, 1, 2))
            rgb_embeds, rgb_grid_embeds = self.rgb_encoder(obs_view12)  

            ''' waypoint prediction ----------------------------- '''
            if waypoint_heatmap_logits_student is not None and waypoint_heatmap_logits_teacher is not None:
                if np.random.rand() > self.keep_origin_view_wp_prob:
                    waypoint_heatmap_logits = waypoint_heatmap_logits_student
                else:
                    waypoint_heatmap_logits = waypoint_heatmap_logits_teacher
            else:
                waypoint_heatmap_logits = self.waypoint_predictor_student(
                    rgb_embeds, depth_embeds)

            # reverse the order of images back to counter-clockwise
            rgb_embeds_reshape = rgb_embeds.reshape(batch_size, NUM_IMGS, 512, 1, 1)
            rgb_feats = torch.cat(
                [rgb_embeds_reshape[:,0:1,:], torch.flip(rgb_embeds_reshape[:,1:,:], [1])], dim=1
            )

            depth_embeds_reshape = depth_embeds.reshape(batch_size, NUM_IMGS, 128, 4, 4)
            depth_feats = torch.cat(
                [depth_embeds_reshape[:,0:1,:], torch.flip(depth_embeds_reshape[:,1:,:], [1])], dim=1
            )

            # grid feats stay clockwise
            rgb_grid_feats = rgb_grid_embeds.reshape(batch_size, NUM_IMGS, 196, 768)
            depth_grid_feats = depth_grid_embeds.reshape(batch_size, NUM_IMGS, 196, 1)

            # from heatmap to points
            batch_x_norm = torch.softmax(
                waypoint_heatmap_logits.reshape(batch_size, NUM_ANGLES*NUM_CLASSES), dim=1
            )
            batch_x_norm = batch_x_norm.reshape(batch_size, NUM_ANGLES, NUM_CLASSES)
            batch_x_norm_wrap = torch.cat((
                batch_x_norm[:,-1:,:], 
                batch_x_norm, 
                batch_x_norm[:,:1,:]), 
                dim=1)
            batch_output_map = nms(
                batch_x_norm_wrap.unsqueeze(1), 
                max_predictions=5,
                sigma=(7.0,5.0))

            # predicted waypoints before sampling
            batch_output_map = batch_output_map.squeeze(1)[:,1:-1,:]

            if in_train:
                # Waypoint augmentation
                # parts of heatmap for sampling (fix offset first)
                HEATMAP_OFFSET = 5
                batch_way_heats_regional = torch.cat(
                    (waypoint_heatmap_logits[:,-HEATMAP_OFFSET:,:], 
                    waypoint_heatmap_logits[:,:-HEATMAP_OFFSET,:],
                ), dim=1)
                batch_way_heats_regional = batch_way_heats_regional.reshape(batch_size, 12, 10, 12)
                batch_sample_angle_idxes = []
                batch_sample_distance_idxes = []
                # batch_way_log_prob = []
                for j in range(batch_size):
                    # angle indexes with candidates
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    # clockwise image indexes (same as batch_x_norm)
                    img_idxes = ((angle_idxes.cpu().numpy()+5) // 10)
                    img_idxes[img_idxes==12] = 0
                    # # candidate waypoint states
                    # way_feats_regional = way_feats[j][img_idxes]
                    # heatmap regions for sampling
                    way_heats_regional = batch_way_heats_regional[j][img_idxes].view(img_idxes.size, -1)
                    way_heats_probs = F.softmax(way_heats_regional, 1)
                    probs_c = torch.distributions.Categorical(way_heats_probs)
                    way_heats_act = probs_c.sample().detach()
                    sample_angle_idxes = []
                    sample_distance_idxes = []
                    for k, way_act in enumerate(way_heats_act):
                        if img_idxes[k] != 0:
                            angle_pointer = (img_idxes[k] - 1) * 10 + 5
                        else:
                            angle_pointer = 0
                        sample_angle_idxes.append(way_act//12+angle_pointer)
                        sample_distance_idxes.append(way_act%12)
                    batch_sample_angle_idxes.append(sample_angle_idxes)
                    batch_sample_distance_idxes.append(sample_distance_idxes)
            
            rgb_feats = self.space_pool_rgb(rgb_feats)          # B x 12 x 512
            depth_feats = self.space_pool_depth(depth_feats)    # B x 12 x 128
            
            rgb_feats = self.rgb_prediction(rgb_feats.view(-1, 512)).view(-1, 12, 512)
            depth_feats = self.depth_prediction(depth_feats.view(-1, 128)).view(-1, 12, 128)

            # for cand
            cand_rgb = []
            cand_depth = []
            cand_angle_fts = []
            cand_img_idxes = []
            cand_angles = []
            cand_distances = []
            for j in range(batch_size):
                if in_train:
                    angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                    distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
                else:
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    distance_idxes = batch_output_map[j].nonzero()[:, 1]
                # for angle & distance
                angle_rad_c = angle_idxes.cpu().float()/120*2*math.pi       # clockwise
                angle_rad_cc = 2*math.pi-angle_idxes.float()/120*2*math.pi  # counter-clockwise
                cand_angle_fts.append( angle_feature_torch(angle_rad_c) )
                cand_angles.append(angle_rad_cc.tolist())
                cand_distances.append( ((distance_idxes + 1)*0.25).tolist() )
                # for img idxes
                img_idxes = 12 - (angle_idxes.cpu().numpy()+5) // 10        # counter-clockwise
                img_idxes[img_idxes==12] = 0
                cand_img_idxes.append(img_idxes)
                # for rgb & depth
                cand_rgb.append(rgb_feats[j, img_idxes, ...])
                cand_depth.append(depth_feats[j, img_idxes, ...])
            
            # for pano
            pano_rgb = rgb_feats                            # B x 12 x 512
            pano_depth = depth_feats                        # B x 12 x 128
            pano_angle_fts = deepcopy(self.pano_angle_fts)  # 12 x 4
            pano_img_idxes = deepcopy(self.pano_img_idxes)  # 12

            # cand_angle_fts 顺时针
            # cand_angles 逆时针
            outputs = {
                'cand_rgb': cand_rgb,               # [K x 512]     
                'cand_depth': cand_depth,           # [K x 128]     
                'cand_angle_fts': cand_angle_fts,   # [K x 4]       
                'cand_img_idxes': cand_img_idxes,   # [K]           
                'cand_angles': cand_angles,         # [K]           
                'cand_distances': cand_distances,   # [K]

                'pano_rgb': pano_rgb,                   # B x 12 x 512          
                'pano_depth': pano_depth,               # B x 12 x 128          
                'pano_rgb_grid': rgb_grid_feats,        # B x 12 x 196 x 768    
                'pano_depth_grid': depth_grid_feats,    # B x 12 x 128 x 1      
                'pano_angle_fts': pano_angle_fts,       # 12 x 4                    
                'pano_img_idxes': pano_img_idxes,       # 12                    
            }
            
            return outputs

        elif mode == 'panorama':
            rgb_fts = self.drop_env(rgb_fts)
            outs = self.vln_bert.forward_panorama(
                rgb_fts, dep_fts, loc_fts, nav_types, view_lens,
            )
            return outs

        elif mode == 'navigation':
            bev_fts = self.drop_env(bev_fts)
            outs = self.vln_bert.forward_navigation(
                txt_embeds, txt_masks, 
                gmap_vp_ids, gmap_step_ids,
                gmap_img_fts, gmap_pos_fts, 
                gmap_masks, gmap_visited_masks, gmap_pair_dists,
                bev_fts, bev_pos_fts, 
                bev_masks, bev_nav_masks, 
                bev_cand_idxs, bev_cand_vpids,
            )
            return outs

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
