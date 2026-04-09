export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name VILETP_rxr
      --run-type train
      --exp-config run_rxr/VILETP.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 8
      IL.iters 20000
      IL.lr 1.5e-5
      IL.log_every 50
      IL.resample_interval 50
      IL.ml_weight 1.0
      IL.cl_weight 0.2
      IL.wpd_weight 8.0
      IL.sample_ratio 0.75
      IL.decay_interval 4000
      IL.load_from_ckpt False
      IL.is_requeue True
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      IL.expert_policy ndtw
      MODEL.keep_origin_view_prob 0.1
      MODEL.keep_origin_view_wp_prob 0.1
      "

flag2=" --exp_name VILETP_rxr
      --run-type eval
      --exp-config run_rxr/VILETP.yaml
      EVAL.SPLIT val_seen
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/VILETP_rxr_gpu=8_env=2_lr=1.5e-5_h=0.5_a=30_p=0.1_wpp=0.1_clwt=0.1_wpdwt=0.0/ckpt.iter17550.pth
      MODEL.keep_origin_view_prob 1.0
      MODEL.keep_origin_view_wp_prob 0.0
      IL.back_algo control
      "


mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag2
      ;;
esac