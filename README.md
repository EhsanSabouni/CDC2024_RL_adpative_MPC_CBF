# Reinforcement Learning-based Adaptive Control Barrier Functions using Receding Horizon Control for Safety-Critical Systems

The repository contains the simulation code for the paper "Reinforcement Learning-based Adaptive Control Barrier Functions using Receding Horizon Control for Safety-Critical Systems" submitted at CDC 2024.

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
To generate the neumerical results shown in the table I, please run the following command.
```
python train_RL_env_MPC_AV.py
```
## License

