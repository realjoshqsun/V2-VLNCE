export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export TORCH_DISTRIBUTED_DEBUG=DETAIL


flag1="--exp_name VILBEV
      --run-type train
      --exp-config run_r2r/VILBEV.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 4
      IL.iters 10000
      IL.decay_interval 2000
      IL.log_every 50
      IL.resample_interval 50
      IL.lr 1e-5
      IL.ml_weight 1.0
      IL.cl_weight 0.2
      IL.wpd_weight 15.0
      IL.sample_ratio 0.75
      IL.is_requeue True
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path data/logs/checkpoints/bev_ckpt/ckpt.iter9600.pth
      MODEL.temperature 1.0
      MODEL.keep_origin_view_prob 0.1
      MODEL.keep_origin_view_wp_prob 0.1
      "

flag2=" --exp_name VILBEV
      --run-type eval
      --exp-config run_r2r/VILBEV.yaml
      TRAINER_NAME eval-vv-sampling-VIL-BEV
      EVAL.SPLIT val_seen
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      IL.iters 10000
      IL.log_every 50
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/VILBEV_resample=50_gpu=4_env=4_lr=1e-5_h=0.5_a=30_p=0.1_wpp=0.1_clwt=0.2_wpdwt=10.0_temp=1_ckpt12000/ckpt.iter4900.pth
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
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag2
      ;;
esac