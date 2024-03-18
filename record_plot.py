# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from utils_folder import utils
import matplotlib.pyplot as plt

class Recorder:
    def __init__(self, root_dir):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_plots'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.colors = ["tab:green", "tab:red", "tab:blue", "tab:orange", 
                        "tab:brown", "tab:pink", "tab:purple", "tab:cyan", "tab:olive"]

    def init(self, env, num_params, num_episodes):
        self.h = env.tau
        self.goal = env.goal_state

        self.masscart = env.masscart
        self.masspole = env.masspole
        self.length = env.length
        self.friction = env.friction

        self.n_steps = env.n_steps

        self.num_params = num_params

        # we consider +-50% interval
        self.masscart_high = env.masscart_high
        self.masscart_low = env.masscart_low 
        self.masspole_high = env.masspole_high
        self.masspole_low = env.masspole_low
        self.length_high = env.length_high
        self.length_low = env.length_low
        self.friction_high = env.friction_high
        self.friction_low = env.friction_low

        self.x =[[] for _ in range(num_episodes)]
        self.x_dot =[[] for _ in range(num_episodes)]
        self.theta =[[] for _ in range(num_episodes)]
        self.theta_dot =[[] for _ in range(num_episodes)]

        self.m = [[] for _ in range(num_episodes)]
        self.M = [[] for _ in range(num_episodes)]
        self.L = [[] for _ in range(num_episodes)]
        self.d = [[] for _ in range(num_episodes)]

        self.u = [[] for _ in range(num_episodes)]

        self.params = [[[] for _ in range(num_episodes)] for _ in range(num_params)]

    def record(self, info, params, episode):

        self.x[episode].append(info["x"])
        self.x_dot[episode].append(info["x_dot"])
        self.theta[episode].append(info["theta"])
        self.theta_dot[episode].append(info["theta_dot"])

        self.m[episode].append(info["masspole"])
        self.M[episode].append(info["masscart"])
        self.L[episode].append(info["length"])
        self.d[episode].append(info["friction"])

        self.u[episode].append(info["control_action"][0])

        for param_ID in range(self.num_params):
            self.params[param_ID][episode].append(params[param_ID])

    def save(self, file_name):
        steps = np.linspace(0, self.n_steps*self.h, self.n_steps)

        x_mean = np.mean(np.array(self.x), 0)
        x_std = np.std(np.array(self.x), 0)
        x_dot_mean = np.mean(np.array(self.x_dot), 0)
        x_dot_std = np.std(np.array(self.x_dot), 0)
        theta_mean = np.mean(np.array(self.theta), 0)
        theta_std = np.std(np.array(self.theta), 0)
        theta_dot_mean = np.mean(np.array(self.theta_dot), 0)
        theta_dot_std = np.std(np.array(self.theta_dot), 0)

        m_mean = np.mean(np.array(self.m), 0)
        m_std = np.std(np.array(self.m), 0)
        M_mean = np.mean(np.array(self.M), 0)
        M_std = np.std(np.array(self.M), 0)
        L_mean = np.mean(np.array(self.L), 0)
        L_std = np.std(np.array(self.L), 0)
        d_mean = np.mean(np.array(self.d), 0)
        d_std = np.std(np.array(self.d), 0)

        u_mean = np.mean(np.array(self.u), 0)
        u_std = np.std(np.array(self.u), 0)

        params_mean = []
        params_std = []

        for i in range(self.num_params):
            params_mean.append(np.mean(np.array(self.params[i][:]), 0))
            params_std.append(np.std(np.array(self.params[i][:]), 0))

        columns = 2
        rows = 3

        fig, ax_array = plt.subplots(rows, columns, figsize=(10,15))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)

        for z, ax_row in enumerate(ax_array):
            for j, axes in enumerate(ax_row):

                if z == 0 and j == 0:
                    axes.plot(steps, x_mean, label = 'x', c = self.colors[0])
                    axes.fill_between(steps, x_mean-x_std, x_mean+x_std, alpha=0.1, facecolor=self.colors[0])
                    axes.grid(linestyle='-', linewidth=0.5)
                    axes.set_facecolor('whitesmoke')
                    axes.plot(steps, self.goal[j]*np.ones((len(steps),)), '--k', label='reference')
                    axes.set_xlabel(r'Time [s]')
                    axes.set_ylabel('cart position')
                    axes.legend(loc='best')

                if z == 0 and j == 1:
                    axes.plot(steps, x_dot_mean, label = 'x_dot', c = self.colors[1])
                    axes.fill_between(steps, x_dot_mean-x_dot_std, x_dot_mean+x_dot_std, alpha=0.1, facecolor=self.colors[1])
                    axes.grid(linestyle='-', linewidth=0.5)
                    axes.set_facecolor('whitesmoke')
                    axes.plot(steps, self.goal[j]*np.ones((len(steps),)), '--k', label='reference')
                    axes.set_xlabel(r'Time [s]')
                    axes.set_ylabel('cart velocity')
                    axes.legend(loc='best')

                if z == 1 and j == 0:
                    axes.plot(steps, theta_mean, label = 'theta', c = self.colors[2])
                    axes.fill_between(steps, theta_mean-theta_std, theta_mean+theta_std, alpha=0.1, facecolor=self.colors[2])
                    axes.grid(linestyle='-', linewidth=0.5)
                    axes.set_facecolor('whitesmoke')
                    axes.plot(steps, self.goal[j]*np.ones((len(steps),)), '--k', label='reference')
                    axes.set_xlabel(r'Time [s]')
                    axes.set_ylabel('pole angle')
                    axes.legend(loc='best')

                if z == 1 and j == 1:
                    axes.plot(steps, theta_dot_mean, label = 'theta_dot', c = self.colors[3])
                    axes.fill_between(steps, theta_dot_mean-theta_dot_std, theta_dot_mean+theta_dot_std, alpha=0.1, facecolor=self.colors[3])
                    axes.grid(linestyle='-', linewidth=0.5)
                    axes.set_facecolor('whitesmoke')
                    axes.plot(steps, self.goal[j]*np.ones((len(steps),)), '--k', label='reference')
                    axes.set_xlabel(r'Time [s]')
                    axes.set_ylabel('pole velocity')
                    axes.legend(loc='best')

                if z == 2 and j == 0:
                    axes.plot(steps, u_mean, label = 'control action', c = self.colors[8])
                    axes.fill_between(steps, u_mean-u_std, u_mean+u_std, alpha=0.1, facecolor=self.colors[8])
                    axes.grid(linestyle='-', linewidth=0.5)
                    axes.set_facecolor('whitesmoke')
                    axes.plot(steps, 0*np.ones((len(steps),)), '--k', label='no control')
                    axes.set_xlabel(r'Time [s]')
                    axes.set_ylabel('u')
                    axes.legend(loc='best')

        path = self.save_dir / file_name
        plt.savefig(str(path)+"_states.pdf", format='pdf', bbox_inches='tight')
        plt.close()


        columns = 2
        rows = 2
        
        fig, ax_array = plt.subplots(rows, columns, figsize=(10,10))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)

        for z, ax_row in enumerate(ax_array):
            for j, axes in enumerate(ax_row):
                if z == 0 and j == 0:
                    axes.plot(steps, m_mean, label = 'masspole', c = self.colors[4])
                    axes.fill_between(steps, m_mean-m_std, m_mean+m_std, alpha=0.1, facecolor=self.colors[4])
                    axes.grid(linestyle='-', linewidth=0.5)
                    axes.set_facecolor('whitesmoke')
                    axes.plot(steps, self.masspole*np.ones((len(steps),)), '--b', label='ground truth')
                    axes.set_xlabel(r'Time [s]')
                    axes.set_ylabel('masspole')
                    axes.set_ylim([self.masspole_low-0.1, self.masspole_high+0.1])
                    axes.legend(loc='best')

                if z == 0 and j == 1:
                    axes.plot(steps, M_mean, label = 'masscart', c = self.colors[5])
                    axes.fill_between(steps, M_mean-M_std, M_mean+M_std, alpha=0.1, facecolor=self.colors[5])
                    axes.grid(linestyle='-', linewidth=0.5)
                    axes.set_facecolor('whitesmoke')
                    axes.plot(steps, self.masscart*np.ones((len(steps),)), '--b', label='ground truth')
                    axes.set_xlabel(r'Time [s]')
                    axes.set_ylabel('masscart')
                    axes.set_ylim([self.masscart_low-0.1, self.masscart_high+0.1])
                    axes.legend(loc='best')

                if z == 1 and j == 0:
                    axes.plot(steps, L_mean, label = 'lengthpole', c = self.colors[6])
                    axes.fill_between(steps, L_mean-L_std, L_mean+L_std, alpha=0.1, facecolor=self.colors[6])
                    axes.grid(linestyle='-', linewidth=0.5)
                    axes.set_facecolor('whitesmoke')
                    axes.plot(steps, self.length*np.ones((len(steps),)), '--b', label='ground truth')
                    axes.set_xlabel(r'Time [s]')
                    axes.set_ylabel('lengthpole')
                    axes.set_ylim([self.length_low-0.1, self.length_high+0.1])
                    axes.legend(loc='best')

                if z == 1 and j == 1:
                    axes.plot(steps, d_mean, label = 'dumping factor', c = self.colors[7])
                    axes.fill_between(steps, d_mean-d_std, d_mean+d_std, alpha=0.1, facecolor=self.colors[7])
                    axes.grid(linestyle='-', linewidth=0.5)
                    axes.set_facecolor('whitesmoke')
                    axes.plot(steps, self.friction*np.ones((len(steps),)), '--b', label='ground truth')
                    axes.set_xlabel(r'Time [s]')
                    axes.set_ylabel('friction')
                    axes.set_ylim([self.friction_low-0.1, self.friction_high+0.1])
                    axes.legend(loc='best')

        path = self.save_dir / file_name
        plt.savefig(str(path)+"_env_parameters.pdf", format='pdf', bbox_inches='tight')
        plt.close()

        columns = 1
        rows = self.num_params

        fig, ax_array = plt.subplots(rows, columns, figsize=(10,3*self.num_params))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)

        k=0
        for z, ax_row in enumerate(ax_array):
            for j, axes in enumerate([ax_row]):

                axes.plot(steps, params_mean[k], label = 'x', c = self.colors[k])
                axes.fill_between(steps, params_mean[k]-params_std[k], params_mean[k]+params_std[k], alpha=0.1, facecolor=self.colors[k])
                axes.grid(linestyle='-', linewidth=0.5)
                axes.set_facecolor('whitesmoke')
                axes.set_xlabel(r'Time [s]')
                axes.set_ylabel(f'param {k}')
                axes.legend(loc='best')

                k+=1

        path = self.save_dir / file_name
        plt.savefig(str(path)+"_QP_parameters.pdf", format='pdf', bbox_inches='tight')
        plt.close()






