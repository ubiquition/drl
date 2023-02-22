# Distributed Proximal Policy Optimization

Distributed Proximal Policy Optimization (DPPO) is a new  distributed architecture which has several GPU trainers and CPU samplers. The data sampled from these samplers are stored in [Redis](https://redis.io/), and these trainers are sharing their network parameters by `share_memory()` in Pytorch, this method is much faster than local and global memory.

## Main requirements

- python == 3.9
- gym==0.24.1
- gym-microrts==0.3.2
- [mujoco](https://github.com/openai/mujoco-py) == 2.2.2
- torch==1.12.0+cu116
- redis==4.3.4
- numba==0.55.2

## Install

Install all requirements.

`pip install -r requirements.txt`

## Running the code

We include two environments ([Mujoco](https://mujoco.org/) and [Microrts](https://github.com/Farama-Foundation/MicroRTS)) and two distributions (normal and beta).

```
│  README.md
│  requirements.txt
│  
├─algo_envs
│  │  algo_base.py
│  │  algo_transformer.py
│  │  ppo_microrts_hogwild.py
│  │  ppo_microrts_share.py
│  │  ppo_microrts_share_gae.py
│  │  ppo_mujoco_beta_hogwild.py
│  │  ppo_mujoco_beta_share.py
│  │  ppo_mujoco_beta_share_gae.py
│  │  ppo_mujoco_normal_hogwild.py
│  │  ppo_mujoco_normal_share.py
│  │  ppo_mujoco_normal_share_gae.py
│  │  __init__.py
│          
├─libs  
│      config.py
│      log.py
│      redis_cache.py
│      redis_config.py
│      utils.py
│      __init__.py
│               
└─train_main_local
        board_start.sh
        board_stop.sh
        checker.py
        mps_start.sh
        mps_stop.sh
        sampler.py
        trainer.py
        train_main_local.py
        train_start.sh
        train_stop.sh
```

You can train them in `train_main_local` or their own files.

**Train example**

```python
python train_main_local/train_main_local.py
```

**Train in their respective files** 

```
python algo_envs/ppo_mujoco_normal_share.py
```

**Where to modify our algorithms or network structure**

You can design your own reinforcement learning through  modifying `Calculate` class (e.g., `PPOMujocoNormalShareCalculate`),  `Calculate` class is mainly used to calculate gradient loss and update network parameter.

Also, the network structure could be modified in `Net` class (e.g., `PPOMujocoNormalShareNet`) which mainly utilized to devise and initialize network structure, output what you want (e.g., state-value of a state, an action you want take and so on).

## papers
https://arxiv.org/abs/2301.10919
https://arxiv.org/abs/2301.10920


