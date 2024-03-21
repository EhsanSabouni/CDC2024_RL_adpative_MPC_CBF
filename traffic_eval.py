import numpy as np
from traffic_env import traffic_env
import matplotlib.pyplot as plt
from control import Control
from Traffic_builder import manual
import torch
from utils_folder import utils
from agents.sac import DiagGaussianActor
import os
from utils_folder.utils import InvalidInputError

class traffic_env_eval():

    def __init__(self, params_range, net_action_space,hidden_dim, total_num_cars, N, dt, agent, render_mode = 'Visualization'):
        self.params_range = params_range
        self.net_action_space = net_action_space
        self.N = N
        self.dt = dt
        self.total_num_cars = total_num_cars
        self.env = traffic_env(total_num_cars, dt, N, render_mode)  # Initilizing a map (object)
        self.actor_model = DiagGaussianActor(agent.obs_dim[0], net_action_space[0], hidden_dim, agent.hidden_depth, agent.log_std_bounds).to('cpu')



    def unsquash(self, value):
        for i in range(0, self.net_action_space[0]):
            low = self.params_range[i][0]
            high = self.params_range[i][1]
            value[i] = ((value[i] + 1) / 2) * (high - low) + low
            value[i] = np.clip(value[i], low, high)
        return value

    def main(self, actor, method, type):

        self.env.car_pointer = 0
        self.env.mapcoordinates() #Extracting the map coordinates
        cars = manual(self.env, self.N, self.total_num_cars)

        que = []
        control = Control(self.env.dt, 'mpc', self.env.N) #Initilizing the controller
        self.env.timeout = 30

        self.env.simtime = np.append(self.env.simtime , self.env.simindex * self.env.dt)

        steps = int(float(self.env.timeout / self.env.dt))

        for self.env.simindex in range(0, steps):
            while (self.env.car_pointer <= self.env.total_num_cars-1 and self.env.simindex == int(cars[self.env.car_pointer].t0 * int(1./self.env.dt))):
                que = np.append(que, cars[self.env.car_pointer])
                self.env.car_pointer = self.env.car_pointer + 1
            leave_ind = []
            for index in range(0, len(que)):

                ego = que[index]
                CurrState = [ego.x, ego.y, ego.psi, ego.v]
                ego = self.env.checkconflicts(ego, que, index)
                obs = CurrState + [ego.acc, ego.steer] + ego.ip_states + ego.ic_states
                x_init = obs
                'RL codes to set the params'

                normalized_observation = utils.normalize(obs, ego.road)


                if method == 'RL_MPC':
                    current_directory = os.path.dirname(__file__)
                    model_path = os.path.join(current_directory, 'saved_actors')
                    model_path = os.path.join(model_path, 'actor_1.pth')

                    self.actor_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    normalized_observation = torch.FloatTensor(normalized_observation).to('cpu')
                    normalized_observation = normalized_observation.unsqueeze(0)
                    dist = self.actor_model(normalized_observation)

                    unsquash_params = dist.mean
                    squashed_params = unsquash_params.clamp(-1, 1)
                    squashed_params = utils.to_np(squashed_params[0])
                    params = self.unsquash(squashed_params)
                elif method == 'baseline':
                    if type == 'c':
                        params = 20 * [0.5]
                    elif type == 'mc':
                        params = 20 * [1]
                    elif type == 'ma':
                        params = 20 * [3]
                    elif type == 'a':
                        params = 20 * [4]
                    else:
                        raise InvalidInputError("Invalid arguments for baseline type")
                        return None


                'end of RL codes'
                'Control codes to calculate the control input'
                Status, action= control.mpc_exec(ego, x_init, params)

                ego.acc, ego.steering = action
                self.env.simindex = self.env.simindex + 1


                'Data gathering'
                self.env.simtime = np.append(self.env.simtime, self.env.simindex * self.env.dt)

                leave_ind = self.env.checkleave(ego, index, leave_ind)

            self.env.step(que)
            if leave_ind:
                # update metric
                self.env.metric_update(que, leave_ind, flag=1)
                que = np.delete(que, [leave_ind])


            self.env.render(que)

        if self.env.render_mode == 1:
            plt.ioff()


        return self.env.ave_time, self.env.ave_fuel, self.env.ave_energy, self.env.metrics



if __name__ == '__main__':
    actor_model = DiagGaussianActor(14, 20, 512, 2, [-10, 10]).to('cpu')
    total_num_cars = 6
    N = 5
    dt = 0.2
    params_range = [[0.1, 5], [0.1, 5], [0.1, 5], [0.1, 5],
               [0.1, 5], [0.1, 5], [0.1, 5], [0.1, 5], [0.1, 5], [0.1, 5],
               [0.1, 5], [0.1, 5], [0.1, 5], [0.1, 5], [0.1, 5], [0.1, 5],
               [0.0001, 1], [0.0001, 1], [0.0001, 1], [0.0001, 1]]

    te = traffic_env_eval(params_range, [20], total_num_cars, N, dt, render_mode = 'Visualization')
    animation = 1
    flag = 1
    ave_time, ave_fuel, ave_energy, metrics = te.main(actor_model,flag)

    fig2, axs2 = plt.subplots(3)
    fig5, axs5 = plt.subplots(2)

    axs2[0].set_title('2D-position')
    axs2[1].set_title('speed')
    axs2[2].set_title('psi')
    axs5[0].set_title('Acceleration')
    axs5[1].set_title('Steering Angle')

    start_pointer = 0
    end_pointer = total_num_cars
    for p in range(start_pointer, end_pointer):
        axs2[0].plot(metrics[p]['pos_x'], metrics[p]['pos_y'], label='car {}'.format(p))
        axs2[1].plot(metrics[p]['time'], metrics[p]['speed'], label='car {}'.format(p))
        axs2[2].plot(metrics[p]['time'], metrics[p]['psi'], label='car {}'.format(p))
        axs5[0].plot(metrics[p]['time'], metrics[p]['acc'], label='car {}'.format(p))
        axs5[1].plot(metrics[p]['time'], metrics[p]['steer'], label='car {}'.format(p))

    axs5[0].legend()
    axs5[1].legend()
    axs2[0].legend()
    axs2[1].legend()
    axs2[2].legend()
    plt.show()
