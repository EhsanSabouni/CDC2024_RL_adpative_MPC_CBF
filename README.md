# Reinforcement Learning-based Adaptive Control Barrier Functions using Receding Horizon Control for Safety-Critical Systems

The repository contains the simulation code for the paper "Reinforcement Learning-based Adaptive Control Barrier Functions using Receding Horizon Control for Safety-Critical Systems" submitted at CDC 2024.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub contributors](https://img.shields.io/github/contributors/username/repository.svg)](https://GitHub.com/username/repository/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/username/repository.svg)](https://GitHub.com/username/repository/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/username/repository.svg)](https://GitHub.com/username/repository/pulls/)
[![GitHub forks](https://img.shields.io/github/forks/username/repository.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/username/repository/network/)
[![GitHub stars](https://img.shields.io/github/stars/username/repository.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/username/repository/stargazers/)

## Table of Contents
- [Compatibility](#compatibility)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)
- [Contributers](#contributors)

## Compatibility
This code has been tested on ubuntu 20.04. Although there should be no difficulty with running it on windows 10+ 
## Installation

Please setup the conda environment by running the following command.

```
conda env create -n envName -f environment.yml
```
Then download/clone the repository.

## Results
This is the video of the simulation results presented in the paper as a concept of proof for feasiblity improvement.
<p align="center">
  <img src="mixed_video.gif">
</p>


To generate the numerical results shown in the table I for the proposed approach, please run the following command.

```
python train_RL_env_MPC_AV.py method=RL_MPC_CBF

```

To generate the results for the baseline case you need specify modes with which you can run the experiments, conservative, moderately conservative, moderately aggressive, aggressive denoted by 'c', mc,'ma','a' in the code.
For example in order to run the baseline-conservative, the following command has to be run:
```
python train_RL_env_MPC_AV.py method=baseline type=c
```

Please note that all the runs can be with or without visualization (the default is with Visualization). To disable visualization the following command can be run: 
```
python train_RL_env_MPC_AV.py render_mode=None
```
## License

### Contributors

Thanks to the following people who have contributed to this project:

- [Sabbir H M Ahmad](https://github.com/SabbirAhmad26)
- [Vittorio Giammarino](https://github.com/VittorioGiammarino)
