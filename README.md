# View Invariant Learning for Vision-Language Navigation in Continuous Environments

[![Home Page](https://img.shields.io/badge/Homepage-V2_VLNCE-144B9E.svg)](https://realjoshqsun.github.io/V2-VLNCE/)
[![arXiv](https://img.shields.io/badge/Arxiv-V2_VLNCE-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.08831v3)

## Core Highlights
🤏 $V^2$-VLNCE Benchmark Integration: Effortlessly extend any standard VLNCE benchmark to the more challenging Varied Viewpoint ($V^2$) scenario with just a few lines of code. No new datasets required. 

🧬 View-Invariant Learning (VIL): A plug-and-play post-training framework that learns sparse, viewpoint-invariant features through a novel contrastive learning objective. 

🎓 Teacher-Student Distillation: Features a specialized distillation framework for the Waypoint Predictor, where a view-dependent teacher guides a student model to maintain robust navigation under drastic camera shifts. 

🚀 State-of-the-Art Performance: Outperforms existing baselines on $V^2$-VLNCE by 8-15% in Success Rate (SR) across R2R-CE and RxR-CE, while also improving performance in standard settings

## Environment Set up

```cmd
conda create -n vil_py3.6 python=3.6.12
conda activate vil_py3.6
```

```cmd
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
```

specify pytorch version with cuda

```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
```

install all other packages

```
pip install -r requirements.txt
```

Install habitat-lab v0.1.7 (from source).

```python
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
python setup.py develop --all
```

### (Optional) Install habitat-sim locally

You might see

```
DISPLAY not detected. For headless systems, compile with --headless for EGL support
```

or

```
GLX context does not support multiple GPUs. Please compile with --headless for multi-gpu support via EGL
```

when training. If this happens, it means the habitat-sim is not installed with headless version and you need to install it manually.

Download from ```https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/habitat-sim-0.1.7-py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2```

```cmd
habitat-sim-0.1.7-py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2
```

### Other ways
Our environment is the same as [ETPNav](https://github.com/MarSaKi/ETPNav). You may also refer to their repository.

## Train and Evaluate

Training and evaluating with 4 GPUs
```cmd
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/VILETP.bash train 2333
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/VILETP.bash eval 2444
CUDA_VISIBLE_DEVICES=0 bash run_r2r/VILETP_eval_sampling.bash eval 2555
```


## Acknowledge

Our implementations are partially inspired by [ETPNav](https://github.com/MarSaKi/ETPNav).

Thanks for their great works!

## Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{11419772,
  author={Sun, Josh Qixuan and Weng, Huaiyuan and Xing, Xiaoying and Yeum, Chul Min and Crowley, Mark},
  journal={IEEE Robotics and Automation Letters}, 
  title={View Invariant Learning for Vision-Language Navigation in Continuous Environments}, 
  year={2026},
  volume={11},
  number={5},
  pages={5861-5868},
  doi={10.1109/LRA.2026.3669785}}


