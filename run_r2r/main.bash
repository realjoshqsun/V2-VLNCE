export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name release_r2r_test_gpu_memory
      --run-type train
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 4
      IL.iters 8000
      IL.log_every 100
      IL.lr 1e-5
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 3000
      IL.is_requeue True
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
      "

flag2=" --exp_name release_r2r_eval_vv_sampling
      --run-type eval
      --exp-config run_r2r/iter_eval_vv_sampling.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      EVAL.SPLIT val_unseen
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
      IL.back_algo control
      "

flag3="--exp_name release_r2r_h=0.8_a=-15
      --run-type inference
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [1,2,3]
      TORCH_GPU_IDS [1,2,3]
      GPU_NUMBERS 3
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
      INFERENCE.PREDICTIONS_FILE preds.json
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
      infer)
      echo "###### infer mode ######"
      python -m torch.distributed.launch --nproc_per_node=3 --master_port $2 run.py $flag3
      ;;
esac
