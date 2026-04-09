export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export TORCH_DISTRIBUTED_DEBUG=DETAIL


flag1="--exp_name VILETP
      --run-type train
      --exp-config run_r2r/VILETP.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 4
      IL.view_sampling height_angle
      IL.max_height_shift 0.25
      IL.max_angle_shift 15
      IL.iters 15000
      IL.decay_interval 3000
      IL.log_every 50
      IL.resample_interval 50
      IL.lr 1e-5
      IL.ml_weight 1.0
      IL.cl_weight 0.1
      IL.wpd_weight 10.0
      IL.sample_ratio 0.75
      IL.is_requeue True
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
      MODEL.temperature 1.0
      MODEL.keep_origin_view_prob 0.5
      MODEL.keep_origin_view_wp_prob 0.1
      "

flag2=" --exp_name VILETP
      --run-type eval
      --exp-config run_r2r/VILETP.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 11
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      IL.iters 15000
      IL.log_every 50
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/VILETP_resample=50_gpu=4_env=4_lr=1e-5_h=0.25_a=15_p=0.5_wpp=0.1_clwt=0.2_wpdwt=10.0_temp=1_ckpt12000/
      MODEL.keep_origin_view_prob 1.0
      MODEL.keep_origin_view_wp_prob 0.0
      IL.back_algo control
      "


mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      python -m torch.distributed.launch --nproc_per_node=3 --master_port $2 run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag2
      ;;
esac