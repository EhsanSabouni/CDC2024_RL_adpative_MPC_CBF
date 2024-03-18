# Copyright (c) VG

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  # specify which GPU(s) to be used

from pathlib import Path
import hydra
import numpy as np

import torch
from utils_folder import utils
from logger_folder.logger import Logger
from buffers.replay_buffer import ReplayBuffer
from traffic_eval import traffic_env_eval
from map import map
from vehicle import Car
from mpcenv import MPCCarEnv

torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, net_action_spec, ctrl_dim, ctrl_horizon_dim, cfg):
    cfg.obs_dim = obs_spec
    cfg.net_action_dim = net_action_spec
    cfg.ctrl_dim = ctrl_dim
    cfg.ctrl_horizon_dim = ctrl_horizon_dim
    return hydra.utils.instantiate(cfg)

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        
        ctrl_horizon_space = (self.train_env.parameters_space.shape[0],)
        ctrl_dim = self.train_env.parameters_space.shape
        self.net_action_space = (ctrl_dim[0],) # 4 weights parameters for the Q-matrix + 1 for the control penalty      
        self.agent = make_agent(self.train_env.full_observation_space_dim, 
                                self.net_action_space, 
                                ctrl_dim,
                                ctrl_horizon_space,
                                self.cfg.agent)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        
    def setup(self):
        coordinates = map(0) # Initilizing a map (object)
        coordinates.mapcoordinates()  # Extracting the map coordinates
        
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        # create target envs and agent
        N = self.cfg.horizon
        dt = self.cfg.sampling_time

        vehicle_train = Car(coordinates,0, 0, self.cfg.road, N, 'bicyclemodel')  # Initilizing a car (object)
        self.train_env = MPCCarEnv(coordinates, vehicle_train, dt, self.cfg.experiment_mode, 0, self.cfg.cbf_type, self.cfg.reward_type, self.cfg.explor_type, render_mode= None,  size=5, controller_type='mpc')
        vehicle_eval = Car(coordinates, 0,0, self.cfg.road, N, 'bicyclemodel')  # Initilizing a car (object)
        self.eval_env = MPCCarEnv(coordinates, vehicle_eval, dt, self.cfg.experiment_mode, 1, self.cfg.cbf_type, self.cfg.reward_type, self.cfg.explor_type, render_mode= None, size=5, controller_type='mpc')
        total_num_cars = 6
        self.te = traffic_env_eval(self.cfg.params_range, self.train_env.parameters_space.shape, total_num_cars, N, dt, None)

        ctrl_horizon_space = (self.train_env.parameters_space.shape[0],)  #need to adjust T
        self.replay_buffer = ReplayBuffer(self.train_env.full_observation_space_dim,
                                          self.train_env.observation_space_dim,
                                          ctrl_horizon_space, int(self.cfg.replay_buffer_size),
                                          self.device)

        # initial MPC setup
        self.nu = self.train_env.action_space.shape[0]
        self.N_BATCH = 1
        self.init_params = [0.3] * self.eval_env.parameters_space.shape[0]
        
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode
    
    def unsquash(self, value):
        for i in range(self.net_action_space[0]):
            low = self.cfg.params_range[i][0]
            high = self.cfg.params_range[i][1]
            value[i] = ((value[i]+1)/2)*(high-low)+low
            value[i] = np.clip(value[i], low, high)

        return value
    
    def eval_MPC(self):
        flag = 0
        ave_time, ave_fuel, ave_u2,_ = self.te.main(self.agent, flag)
        with self.logger.log_and_dump_ctx(self.global_step, ty='traffic_eval') as log:
            log('ave_time', ave_time)
            log('ave_u2', ave_u2)
            log('ave_fuel', ave_fuel)


    def eval(self):
        step = 0
        episode = 0
        total_reward = 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            observation,_ = self.eval_env.reset()
            observation = observation['agent']
            self.agent.reset()
            Status = "solution found"
            violation_flag = False
            timeout_flag= False
            done = False
            reward = torch.tensor(0.)
            # take env step
            discount = 0.99

            while not done and not violation_flag and Status == "solution found" and not timeout_flag:

                with torch.no_grad(), utils.eval_mode(self.agent):
                    normalized_observation = utils.normalize(observation,  self.eval_env.agent.road)
                    params_squashed = self.agent.act(normalized_observation, self.global_step, eval_mode=True)
                    params = self.unsquash(params_squashed) #unsquash params


                state = observation.copy()
                # compute action based on current state, dynamics, and cost
                Status, action = self.eval_env.exec(state[0:4], params)
                # MPC end

                # dynamic step
                observation, reward_temp, done,  info = self.eval_env.step(action, params,step)
                violation_flag = info["violation"]
                timeout_flag = info["timeout"]

                self.eval_env.render()

                reward += reward_temp *discount
                discount *= self.agent.discount
                step += 1


            total_reward += reward
            episode += 1
            
        with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_states, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_states, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_states, self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        observation, _ = self.train_env.reset()
        observation = observation['agent']
        metrics = None
        Status = "solution found"
        violation_flag = False
        timeout_flag = False
        done = False    
        discount = 0.99
        reward = torch.tensor(0.)

        current_directory = os.getcwd()
        model_path = os.path.join(current_directory, 'actor.pth')
        while train_until_step(self.global_step):
            if done or Status != "solution found" or violation_flag or timeout_flag:
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_step)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)

                # reset env
                observation,_ = self.train_env.reset()
                observation = observation['agent']

                episode_step = 0
                episode_reward = 0

                # take env step
                reward = torch.tensor(0.) 
                discount = 0.99  

            # try to evaluate
            if eval_every_step(self.global_step):
                self.eval()

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()


            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_buffer, self.global_step)
                self.logger.log_metrics(metrics, self.global_step, ty='train')

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                normalized_observation = utils.normalize(observation, self.train_env.agent.road)
                params_squashed = self.agent.act(normalized_observation, self.global_step, eval_mode=False)
                params = self.unsquash(params_squashed)

            # MPC start here
            state = observation.copy()

            # compute action based on current state, dynamics, and cost
            Status, action = self.train_env.exec(state[0:4], params)
            # MPC end

            # dynamic step
            next_obs, reward_temp, done,  info = self.train_env.step(action, params, episode_step)
            reward += reward_temp *discount
            discount *= self.agent.discount
            violation_flag = info["violation"]
            timeout_flag = info["timeout"]

            #next_state = self.train_env._agent_states.copy()
            next_state = next_obs.copy()
            done_no_max = 0 if episode_step + 1 == self.train_env.n_steps else float(done)
            episode_reward = reward
            normalized_observation = utils.normalize(observation, self.train_env.agent.road)
            normalized_next_obs = utils.normalize(next_obs, self.train_env.agent.road)
            if episode_step > 0:
                self.replay_buffer.add(state, normalized_observation, params_squashed, reward_temp, next_state, normalized_next_obs, float(done), done_no_max)

            observation = next_obs
            episode_step += 1
            self._global_step += 1

            self.train_env.render()
            if self.global_step % self.cfg.save_Freq == 0:
                torch.save(self.agent.actor.state_dict(), model_path)
                self.eval_MPC()
            
    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.cfg.task_name}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.cfg.task_name}.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

@hydra.main(config_path='config_folder', config_name='config_ac_av')
def main(cfg):
    from train_RL_env_MPC_AV import Workspace as W
    workspace = W(cfg)
    workspace.train()

if __name__ == '__main__':
    main()
