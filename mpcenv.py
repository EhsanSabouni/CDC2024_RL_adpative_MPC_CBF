import gym
from gym import spaces
import pygame
import numpy as np
import casadi as cd
from scipy.integrate import odeint,solve_ivp,ode
import torch
import numpy as np
from circle_fit import taubinSVD
import matplotlib.pyplot as plt
import time
from vehicle import Car
import cvxpy as cp
import random
class MPCCarEnv(gym.Env):
    metadata = {"render_modes": ["Visualization"], "render_fps": 4}

    def __init__(self, coordinates, vehicle ,  dt, mode, evalmod, cbf_type, reward_type, exploration_type, render_mode= None, size=5, controller_type = 'qp'):
        self.size = size  # The size of the square grid
        self.dt = dt
        self.mode = mode
        self.type = controller_type
        self.agent = vehicle
        self.evalmode = evalmod
        self.cbf_type = cbf_type
        self.reward_type = reward_type
        if exploration_type == 0:
            self.constant = 0
        elif exploration_type == 1:
            self.constant = 10
        else:
            self.constant = 25

        self.mainroad_agent = Car(coordinates,0.39,  0.39, 0, 1, 'bicyclemodel')  # Initilizing a car (object)
        self.mainroad_agent.set_states(-100, 0,0, 0)
        self.mergingroad_agent = Car(coordinates, 0.39, 0.39, 1, 1, 'bicyclemodel')  # Initilizing a car (object)
        self.mergingroad_agent.set_states(-100, 0, 0, 0)


        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14, 1), dtype=np.float32)
        self.observation_shape = (14,)


        self.window_shape = (140, 160, 1)
        self.observation_space_dim = self.observation_shape 
        self.full_observation_space_dim = self.observation_space_dim 
        self.parameters_space = spaces.Box(low=0, high=1, shape= (20, 1), dtype=np.float32)
        self.coordinates = coordinates
        self.n_steps = 500
        self.endpoint = False
        # We have 2 actions, corresponding acceleration and steering
        self.action_space = spaces.Box(low=np.array([self.agent.umin, self.agent.steermin]), high=np.array([self.agent.umax, self.agent.steermax]), dtype=np.float32)
        self._agent_states = self.agent.get_states()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode == "Visualization":
            plt.style.use('seaborn-v0_8')
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8])

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _get_obs(self):
        states = []
        if self.agent.road == 0:
            states = self.mainroad_agent.get_states() + self.mergingroad_agent.get_states()
        else:
            states = self.mergingroad_agent.get_states() + self.mainroad_agent.get_states()

        return {"agent": self._agent_states + [self.agent.acc, self.agent.steer] + states}

    def _get_info(self):
        pass


    def reset(self):

        V_0 = 1
        if self.evalmode == 0 and self.render_mode == "Visualization":
            self.agent.RightCBF = []
            self.agent.LeftCBF = []
            self.agent.MergingCBF = []
            self.agent.RearendCBF = []
            self.agent.RightOrg = []
            self.agent.LeftOrg = []
            self.agent.MergingOrg = []
            self.agent.RearendOrg = []
            self.agent.accdata = []
            self.agent.steerdata = []
        self.mode = random.randint(0, 3)
        self.agent.road = random.randint(0, 1)
        while True:
            self.mainroad_agent.set_states(-100, 0, 0, 15)
            self.mergingroad_agent.set_states(-100, 0, 0, 15)
            if self.evalmode == 0:
                if self.agent.road == 0:
                    random_number = random.randint(0, 300)
                    initial_V = V_0 + np.random.uniform(0, 14)
                    self.agent.set_states(
                        self.coordinates.mainroad['X_C'][random_number] + np.random.uniform(-2.25, 2.25),
                        self.coordinates.mainroad['Y_C'][random_number] + np.random.uniform(-2.25, 2.25),
                        self.coordinates.mainroad['Psi_Int'][random_number] + 0.2 * np.random.uniform(-1, 1),
                        initial_V)
                    if self.mode == 1 or self.mode == 3:
                        random_number += self.constant
                        random_index = min((random_number + random.randint(int(initial_V * 1.2 / 0.3), 60)), 300)
                        self.mainroad_agent.desired_speed = np.random.uniform(5, 15)
                        self.mainroad_agent.set_states(
                            self.coordinates.mainroad['X_C'][random_index] + np.random.uniform(-1, 1),
                            self.coordinates.mainroad['Y_C'][random_index] + np.random.uniform(-1, 1),
                            self.coordinates.mainroad['Psi_Int'][random_index] + 0.1 * np.random.uniform(-1, 1),
                            V_0 + np.random.uniform(0, 14))
                    if self.mode == 2 or self.mode == 3:
                        random_number += self.constant
                        self.mergingroad_agent.desired_speed = np.random.uniform(5, 15)
                        random_index = min(
                            (random_number + random.randint(int(initial_V * 1.2 * (random_number / 300) / 0.3), 60)),
                            300)
                        self.mergingroad_agent.set_states(
                            self.coordinates.mergingroad['X_C'][random_index] + np.random.uniform(-1, 1),
                            self.coordinates.mergingroad['Y_C'][random_index] + np.random.uniform(-1, 1),
                            self.coordinates.mergingroad['Psi_Int'][random_index] + 0.1 * np.random.uniform(-1, 1),
                            V_0 + np.random.uniform(0, 14))
                else:
                    initial_V = V_0 + np.random.uniform(0, 14)
                    random_number = random.randint(0, 300)
                    self.agent.set_states(
                        self.coordinates.mergingroad['X_C'][random_number] + np.random.uniform(-2.25, 2.25),
                        self.coordinates.mergingroad['Y_C'][random_number] + np.random.uniform(-2.25, 2.25),
                        self.coordinates.mergingroad['Psi_Int'][random_number] + 0.2 * np.random.uniform(-1, 1),
                        initial_V)
                    if self.mode == 1 or self.mode == 3:
                        random_number += self.constant
                        random_index = min((random_number + random.randint(int(initial_V * 1.2 / 0.3), 60)), 300)
                        self.mergingroad_agent.desired_speed = np.random.uniform(5, 15)
                        self.mergingroad_agent.set_states(
                            self.coordinates.mergingroad['X_C'][random_index] + np.random.uniform(-1, 1),
                            self.coordinates.mergingroad['Y_C'][random_index] + np.random.uniform(-1, 1),
                            self.coordinates.mergingroad['Psi_Int'][random_index] + 0.1 * np.random.uniform(-1, 1),
                            V_0 + np.random.uniform(0, 14))
                    if self.mode == 2 or self.mode == 3:
                        random_number += self.constant
                        random_index = min((random_number + random.randint(int(initial_V * 1.2 / 0.3), 60)), 300)
                        self.mainroad_agent.desired_speed = np.random.uniform(5, 15)
                        self.mainroad_agent.set_states(
                            self.coordinates.mainroad['X_C'][random_index] + np.random.uniform(-1, 1),
                            self.coordinates.mainroad['Y_C'][random_index] + np.random.uniform(-1, 1),
                            self.coordinates.mainroad['Psi_Int'][random_index] + 0.1 * np.random.uniform(-1, 1),
                            V_0 + np.random.uniform(0, 14))
            else:
                random_number = random.randint(0, 5)
                if self.agent.road == 0:
                    self.agent.set_states(self.coordinates.mainroad['X_C'][random_number],
                                          self.coordinates.mainroad['Y_C'][random_number],
                                          self.coordinates.mainroad['Psi_Int'][random_number],
                                          np.random.uniform(12, 15))
                    if self.mode == 1 or self.mode == 3:
                        self.mainroad_agent.desired_speed = np.random.uniform(7, 10)
                        random_index = min((random_number + 75), 300)
                        self.mainroad_agent.set_states(self.coordinates.mainroad['X_C'][random_index],
                                                       self.coordinates.mainroad['Y_C'][random_index],
                                                       self.coordinates.mainroad['Psi_Int'][random_index],
                                                       10)
                    if self.mode == 2 or self.mode == 3:
                        self.mergingroad_agent.desired_speed = np.random.uniform(7, 10)
                        random_index = min((random_number + 75), 300)
                        self.mergingroad_agent.set_states(self.coordinates.mergingroad['X_C'][random_index],
                                                          self.coordinates.mergingroad['Y_C'][random_index],
                                                          self.coordinates.mergingroad['Psi_Int'][random_index],
                                                          10)
                else:
                    self.agent.set_states(self.coordinates.mergingroad['X_C'][random_number],
                                          self.coordinates.mergingroad['Y_C'][random_number],
                                          self.coordinates.mergingroad['Psi_Int'][random_number],
                                          np.random.uniform(12, 15))
                    if self.mode == 1 or self.mode == 3:
                        random_index = min((random_number + 75), 300)
                        self.mergingroad_agent.desired_speed = np.random.uniform(7, 10)
                        self.mergingroad_agent.set_states(self.coordinates.mergingroad['X_C'][random_index],
                                                          self.coordinates.mergingroad['Y_C'][random_index],
                                                          self.coordinates.mergingroad['Psi_Int'][random_index],
                                                          10)
                    if self.mode == 2 or self.mode == 3:
                        random_index = min((random_number + 75), 300)
                        self.mainroad_agent.desired_speed = np.random.uniform(7, 10)
                        self.mainroad_agent.set_states(self.coordinates.mainroad['X_C'][random_index],
                                                       self.coordinates.mainroad['Y_C'][random_index],
                                                       self.coordinates.mainroad['Psi_Int'][random_index],
                                                       10)

            self._agent_states = self.agent.get_states()
            if not self.mergingisfeasible() and not self.rearendisfeasible():
                break

        observation = self._get_obs()
        info = []

        return observation, info


    def rearendisfeasible(self):
        x0, y0, psi0, v0 = self.agent.get_states()
        if self.agent.road == 0:
            a = 1.5
            b = 1
            states = self.mainroad_agent.get_states()
            xip, yip, psi_ip, vip = states
            b3 = (np.cos(psi0) * (x0 - xip) - np.sin(psi0) * (y0 - yip)) ** 2 / a + (
                    np.cos(psi0) * (y0 - yip) + np.sin(psi0) * (x0 - xip)) ** 2 / b - v0 ** 2
        else:
            states = self.mergingroad_agent.get_states()
            a = 1.5
            b = 1
            xip, yip, psi_ip, vip = states
            b3 = (np.cos(psi0) * (x0 - xip) - np.sin(psi0) * (y0 - yip)) ** 2 / a + (
                    np.cos(psi0) * (y0 - yip) + np.sin(psi0) * (x0 - xip)) ** 2 / b - v0 ** 2

        return b3 < 0

    def mergingisfeasible(self):

        x0, y0, psi0, v0 = self.agent.get_states()
        if self.agent.road == 0:
            states = self.mergingroad_agent.get_states()
            xic, yic, psi_ic, vic = states
            if xic >= 0:
                R_i = cd.sqrt(x0 ** 2 + (y0 - 1000) ** 2)
                R_c = cd.sqrt(xic ** 2 + (yic - 100) ** 2)
                b01 = 1000
                b00 = 100
                delta = 0
                theta_0_i = -1.4672603502370494
                theta_0_ic = -0.51
                const = 1.52 - 1.4672603502370494
                b4 = (R_c * (theta_0_ic - np.arctan((yic - b00) / xic)) - R_i * (
                        theta_0_i - np.arctan((y0 - b01) / x0))
                      - (theta_0_i - np.arctan((y0 - b01) / x0)) / const * v0 - delta)
            else:
                R_i = cd.sqrt(x0 ** 2 + (y0 - 1000) ** 2)
                b01 = 1000
                delta = 0
                theta_0_i = -1.4672603502370494
                const = 1.52 - 1.4672603502370494
                b4 = 103 - xic - R_i * (
                        theta_0_i - np.arctan((y0 - b01) / x0))\
                      - (theta_0_i - np.arctan((y0 - b01) / x0)) / const * v0 - delta

        else:
            states = self.mainroad_agent.get_states()
            xic, yic, psi_ic, vic = states
            if xic >= 0:
                R_i = cd.sqrt(x0 ** 2 + (y0 - 100) ** 2)
                R_c = cd.sqrt(x0 ** 2 + (y0 - 1000) ** 2)
                b01 = 100
                b00 = 1000
                delta = 0
                theta_0_i = -0.51
                theta_0_ic = -1.4672603502370494
                const = 1.52 - 0.51
                b4 = (R_c * (theta_0_ic - np.arctan((yic - b00) / xic)) - R_i * (
                        theta_0_i - np.arctan((y0 - b01) / x0))
                      - (theta_0_i - np.arctan((y0 - b01) / x0)) / const * v0 - delta)
            else:
                R_i = cd.sqrt(x0 ** 2 + (y0 - 100) ** 2)
                R_c = cd.sqrt(x0 ** 2 + (y0 - 1000) ** 2)
                b01 = 100
                b00 = 1000
                delta = 0
                theta_0_i = -0.51
                theta_0_ic = -1.4672603502370494
                const = 1.52 - 0.51
                b4 = (103-xic - R_i * (
                        theta_0_i - np.arctan((y0 - b01) / x0))
                      - (theta_0_i - np.arctan((y0 - b01) / x0)) / const * v0 - delta)

        return b4 < 0

    def dynamics(self,t,x):

        dx = [0] * 6
        dx[0] = x[3] * np.cos(x[2])
        dx[1] = x[3] * np.sin(x[2])
        dx[2] = (x[3] * x[5])/(0.39 + 0.39)
        dx[3] = x[4]
        dx[4] = 0
        dx[5] = 0

        return dx

    def RoadViolation(self):
        y0 = self._agent_states[1]
        x0 = self._agent_states[0]

            
        xcr, ycr, Rr, sigma = taubinSVD(self.agent.Ref_Rightlane)
        xcl, ycl, Rl, sigma = taubinSVD(self.agent.Ref_Leftlane)
        violation = True

        if Rr >= 1200 or Rl >= 1200:
            if y0 >= self.agent.Ref_Rightlane[0][1] and y0 <= self.agent.Ref_Leftlane[0][1]:
                violation = False
            if y0 <= self.agent.Ref_Rightlane[0][1] and y0 >= self.agent.Ref_Leftlane[0][1]:
                violation = False
        else:
            if ((x0 - xcr) ** 2 + (y0 - ycr) ** 2 <= Rr ** 2) and Rl ** 2 <= (x0 - xcl) ** 2 + (y0 - ycl) ** 2:
                violation = False
            if (x0 - xcr) ** 2 + (y0 - ycr) ** 2 >= Rr ** 2 and Rl ** 2 >= (x0 - xcl) ** 2 + (y0 - ycl) ** 2:
                violation = False


        return violation

    def motion(self, dynamics, states, action, timespan, teval):
        y0 = np.append(states, action)
        sol = solve_ivp(dynamics, timespan, y0, method = 'DOP853', t_eval=[teval], atol=1e-6)
        x = np.reshape(sol.y[0:len(self._agent_states)], len(self._agent_states))
        return x


    def step(self, action, params, count):
        self.endpoint = False

        if self.agent.road == 0:
            states = self.mainroad_agent.get_states()
            mainroad_agent = self.mainroad_agent
            if self.mode == 1 or self.mode == 3:
                Status, main_road_agent_action = self.control_other_agent(states, mainroad_agent)
                states = self.motion(self.dynamics, states, main_road_agent_action, [0, self.dt], self.dt)
                self.mainroad_agent.set_states(states[0], states[1], states[2], states[3])
            states = self.mergingroad_agent.get_states()
            mergingroad_agent = self.mergingroad_agent
            if self.mode == 2 or self.mode == 3:
                Status, mergingroad_agent_action = self.control_other_agent(states, mergingroad_agent)
                states = self.motion(self.dynamics, states, mergingroad_agent_action, [0, self.dt], self.dt)
                self.mergingroad_agent.set_states(states[0], states[1], states[2], states[3])
        else:
            states = self.mergingroad_agent.get_states()
            mergingroad_agent = self.mergingroad_agent
            if self.mode == 1 or self.mode == 3:
                Status, mergingroad_agent_action = self.control_other_agent(states, mergingroad_agent)
                states = self.motion(self.dynamics, states, mergingroad_agent_action, [0, self.dt], self.dt)
                self.mergingroad_agent.set_states(states[0], states[1], states[2], states[3])
            states = self.mainroad_agent.get_states()
            mainroad_agent = self.mainroad_agent
            if self.mode == 2 or self.mode == 3:
                Status, main_road_agent_action = self.control_other_agent(states, mainroad_agent)
                states = self.motion(self.dynamics, states, main_road_agent_action, [0, self.dt], self.dt)
                self.mainroad_agent.set_states(states[0], states[1], states[2], states[3])

        self._agent_states = self.motion(self.dynamics, self._agent_states, action, [0, self.dt], self.dt)
        self.agent.set_states(self._agent_states[0], self._agent_states[1] , self._agent_states[2], self._agent_states[3])
        self.agent.acc = action[0]
        self.agent.steering = action[1]


        if self.agent.road == 0:
            X_C = self.coordinates.mainroad['X_C']
            Y_C = self.coordinates.mainroad['Y_C']
        else:
            X_C = self.coordinates.mergingroad['X_C']
            Y_C = self.coordinates.mergingroad['Y_C']

        Dist2Centerarray = np.array([])
        for k in range(0, len(X_C)):
            Dist2Centerarray = np.append(Dist2Centerarray, (self.agent.x - X_C[k]) ** 2 + (self.agent.y - Y_C[k]) ** 2)
        Index_Point_on_centerlane = np.argmin(Dist2Centerarray)


        self.agent.referencegenerator()
        Ref_Centerlane = self.agent.Ref_Centerlane
        Psi_ref = self.agent.Ref_Psi

        Ref_Centerlane_x = [Ref_Centerlane[i][0] for i in range(0, len(Ref_Centerlane))]
        Ref_Centerlane_y = [Ref_Centerlane[i][1] for i in range(0, len(Ref_Centerlane))]
        waypoints2centerx = [self._agent_states[0]] + Ref_Centerlane_x[2:8]
        waypointstraj2centery = [self._agent_states[1]] + Ref_Centerlane_y[2:8]

        x1 = np.linspace(waypoints2centerx[0], waypoints2centerx[-1], len(Psi_ref))
        y1 = np.linspace(waypointstraj2centery[0], waypointstraj2centery[-1], len(Psi_ref))
        Psi_ref_path = [np.arctan2((y1[i + 1] - y1[i]), (x1[i + 1] - x1[i])) for i in range(len(Psi_ref) - 1)]
        Psi_ref_path_continous = [(psi_ref-self.agent.last_psi_ref_path + np.pi)%(2*np.pi)-np.pi + self.agent.last_psi_ref_path for psi_ref in Psi_ref_path]
        self.agent.last_psi_ref_path = Psi_ref[0]

        reward = torch.tensor(0.)
        v_des = 15
        if self.reward_type == 0:
            b = [0.1569, 0.02450, -0.0007415, 0.00005975]
            c = [0.07224, 0.09681, 0.001075]
            fuel = 0.2 * (action[0] * (c[0] + c[1]*self.agent.v  + c[2]*self.agent.v **2)
                          +(b[0] + b[1]*self.agent.v + b[2]*self.agent.v**2 + b[3]*self.agent.v **3))
            if action[0] >= 0:
                reward += - 0.5 * fuel/1.52
            reward += -0.15 * (self.agent.v - v_des) ** 2 / v_des ** 2
            reward += -0.35 * (self.agent.psi - Psi_ref_path_continous[0]) ** 2 / 3.14 ** 2

        elif self.reward_type == 1:
            reward += -0.25 * (action[0] - 0) ** 2 / 5 **2
            reward += -0.2 * (action[1] - 0) ** 2 / 0.45 ** 2
            reward += -0.25 * (self.agent.v - v_des) ** 2 / v_des ** 2
            reward += -0.3 * (self.agent.psi - Psi_ref_path_continous[0]) ** 2 / 3.14 ** 2

        elif self.reward_type == 2:
            reward += -0.05 * (action[0] - 0) ** 2 / 5 ** 2
            reward += -0.05 * (action[1] - 0) ** 2 / 0.45 ** 2
            reward += -0.45 * (self.agent.v - v_des) ** 2 / v_des ** 2
            reward += -0.45 * (self.agent.psi - Psi_ref_path_continous[0]) ** 2 / 3.14 ** 2

        elif self.reward_type == 3:
            reward += -0.1 * (action[0] - 0) ** 2 / 5 ** 2
            reward += -0.1 * (action[1] - 0) ** 2 / 0.45 ** 2
            reward += -0.4 * (self.agent.v - min(v_des, min(mainroad_agent.v + 5, mergingroad_agent.v + 5))) ** 2 / v_des ** 2
            reward += -0.4 * (self.agent.psi - Psi_ref_path_continous[0]) ** 2 / 3.14 ** 2

        done = False
        thr = 30
        Dist2Goal = (self.agent.x - X_C[-55]) ** 2 + (self.agent.y - Y_C[-55]) ** 2  # distance to the distination array
        if Dist2Goal < thr:
            done = True
            self.endpoint = True
        info = {"violation": self.RoadViolation(), "timeout": False}
        if count > self.n_steps:
            info["timeout"] = True    
        
        if info["violation"] == True or action[0] == -5.66 or info["timeout"] == True: #Violation or infeasibility
            reward = reward - 500 - (1 - Index_Point_on_centerlane/len(Dist2Centerarray)) * 100 # Incentivizing moving further
        if self.agent.road == 0:
            env_state = list(self._agent_states) + list(action) + list(self.mainroad_agent.get_states())+ list(
                self.mergingroad_agent.get_states())
        else:
            env_state = list(self._agent_states) + list(action) + list(self.mergingroad_agent.get_states()) + list(
                self.mainroad_agent.get_states())

        return env_state, reward, done, info

    def exec(self, x_init, params):
        self.agent.referencegenerator()
        Ref_Rightlane = self.agent.Ref_Rightlane
        Ref_Leftlane = self.agent.Ref_Leftlane
        Ref_Centerlane = self.agent.Ref_Centerlane
        Psi_ref = self.agent.Ref_Psi

        x0 = x_init[0]
        y0 = x_init[1]
        psi0 = x_init[2]
        v0 = x_init[3]
        opti = cd.Opti()
        X = opti.variable(self.agent.statesnumber, self.agent.N + 1)
        u = opti.variable(self.agent.Inputnumber + 2, self.agent.N)
        s = opti.variable(4, self.agent.N)

        xcr, ycr, Rr, sigma = taubinSVD(Ref_Rightlane)
        xcl, ycl, Rl, sigma = taubinSVD(Ref_Leftlane)

        Ref_Centerlane_x = [Ref_Centerlane[i][0] for i in range(0, len(Ref_Centerlane))]
        Ref_Centerlane_y = [Ref_Centerlane[i][1] for i in range(0, len(Ref_Centerlane))]
        waypoints2centerx = [x0] + Ref_Centerlane_x[2:8]
        waypointstraj2centery = [y0] + Ref_Centerlane_y[2:8]

        x1 = np.linspace(waypoints2centerx[0], waypoints2centerx[-1], len(Psi_ref))
        y1 = np.linspace(waypointstraj2centery[0], waypointstraj2centery[-1], len(Psi_ref))
        Psi_ref_path = [np.arctan2((y1[i + 1] - y1[i]), (x1[i + 1] - x1[i])) for i in range(len(Psi_ref) - 1)]
        Psi_ref_path_continous = [
            (psi_ref - self.agent.last_psi_ref_path + np.pi) % (2 * np.pi) - np.pi + self.agent.last_psi_ref_path
            for psi_ref in Psi_ref_path]
        self.agent.last_psi_ref_path = Psi_ref[0]
        for k in range(0, self.agent.N):
            if (Rr >= 1200 or Rl >= 1200) or abs(np.sin(Psi_ref[0])) <= 0.05:
                if y0 - Ref_Rightlane[0][1] >= 0:
                    b1 = X[1, k] - Ref_Rightlane[0][1]
                    lfb1 = X[3, k] * cd.sin(X[2, k]) + params[0] * b1
                    l2fb1 = params[0] * (lfb1 - params[0] * b1) + params[1] * lfb1
                    lgu1 = cd.sin(X[2, k])
                    lgdelta1 = X[3, k] * cd.cos(X[2, k]) * (X[3, k]) / (self.agent.lf + self.agent.lr)
                    opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[0, k] >= 0)

                    b2 = -X[1, k] + Ref_Leftlane[0][1]
                    lfb2 = -X[3, k] * cd.sin(X[2, k]) + params[2] * b2
                    l2fb2 = params[2] * (lfb2 - params[2] * b2) + params[3] * lfb2
                    lgu2 = -cd.sin(X[2, k])
                    lgdelta2 = -X[3, k] * cd.cos(X[2, k]) * (X[3, k]) / (self.agent.lf + self.agent.lr)
                    opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] + s[1, k] >= 0)

                else:
                    b1 = X[1, k] - Ref_Leftlane[0][1]
                    lfb1 = X[3, k] * cd.sin(X[2, k]) + params[4] * b1
                    l2fb1 = params[4] * (lfb1 - params[4] * b1) + params[5] * lfb1
                    lgu1 = cd.sin(X[2, k])
                    lgdelta1 = X[3, k] * cd.cos(X[2, k]) * (X[3, k]) / (self.agent.lf + self.agent.lr)
                    opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[0, k] >= 0)

                    b2 = -X[1, k] + Ref_Rightlane[0][1]
                    lfb2 = -X[3, k] * cd.sin(X[2, k]) + params[6] * b2
                    l2fb2 = params[6] * (lfb2 - params[6] * b2) + params[7] * lfb2
                    lgu2 = -cd.sin(X[2, k])
                    lgdelta2 = -X[3, k] * cd.cos(X[2, k]) * (X[3, k]) / (self.agent.lf + self.agent.lr)
                    opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] + s[1, k] >= 0)


            else:
                b1 = cd.sqrt((X[0, k] - xcr) ** 2 + (X[1, k] - ycr) ** 2) - Rr

                lfb1 = (X[3, k] * (X[0, k] * cd.cos(X[2, k]) - xcr * cd.cos(X[2, k]) + X[1, k] * cd.sin(
                    X[2, k]) - ycr * cd.sin(X[2, k]))) / cd.sqrt(
                    (X[0, k] - xcr) ** 2 + (X[1, k] - ycr) ** 2) - \
                       params[8] * (Rr - cd.sqrt((X[0, k] - xcr) ** 2 + (X[1, k] - ycr) ** 2))

                l2fb1 = X[3, k] * cd.cos(X[2, k]) * ((params[8] * (X[0, k] - xcr)) / cd.sqrt(
                    (X[0, k] - xcr) ** 2 + (X[1, k] - ycr) ** 2) + (
                                                             X[3, k] * cd.cos(X[2, k])) / cd.sqrt(
                    (X[0, k] - xcr) ** 2 +
                    (X[1, k] - ycr) ** 2) - (X[3, k] * (2 * X[0, k] - 2 * xcr) * (
                        X[0, k] * cd.cos(X[2, k]) - xcr * cd.cos(X[2, k]) + X[1, k] * cd.sin(
                    X[2, k]) - ycr * cd.sin(X[2, k]))) / (
                                                             2 * ((X[0, k] - xcr) ** 2 +
                                                                  (X[1, k] - ycr) ** 2) ** (3 / 2))) - \
                        params[9] * (
                                params[8] * (Rr - cd.sqrt((X[0, k] - xcr) ** 2 + (X[1, k] - ycr) ** 2)) - (
                                X[3, k] * (X[0, k] * cd.cos(X[2, k]) - xcr * cd.cos(X[2, k]) +
                                           X[1, k] * cd.sin(X[2, k]) - ycr * cd.sin(
                                    X[2, k]))) / cd.sqrt(
                            (X[0, k] - xcr) ** 2 + (X[1, k] - ycr) ** 2)) + X[3, k] * cd.sin(
                    X[2, k]) * (
                                (params[8] * (X[1, k] - ycr)) / cd.sqrt((X[0, k] - xcr) ** 2 +
                                                                        (X[1, k] - ycr) ** 2) + (
                                        X[3, k] * cd.sin(X[2, k])) / cd.sqrt(
                            (X[0, k] - xcr) ** 2 + (X[1, k] - ycr) ** 2) - (
                                        X[3, k] * (2 * X[1, k] - 2 * ycr) * (
                                        X[0, k] * cd.cos(X[2, k]) -
                                        xcr * cd.cos(
                                    X[2, k]) + X[1, k] * cd.sin(X[2, k]) - ycr * cd.sin(
                                    X[2, k]))) / (
                                        2 * ((X[0, k] - xcr) ** 2 + (X[1, k] - ycr) ** 2) ** (3 / 2)))

                lgu1 = (X[0, k] * cd.cos(X[2, k]) - xcr * cd.cos(X[2, k]) + X[1, k] * cd.sin(
                    X[2, k]) - ycr * cd.sin(X[2, k])) / cd.sqrt(
                    (X[0, k] - xcr) ** 2 + (X[1, k] - ycr) ** 2)

                lgdelta1 = (X[3, k] ** 2 * (
                        X[1, k] * cd.cos(X[2, k]) - ycr * cd.cos(X[2, k]) - X[0, k] * cd.sin(
                    X[2, k]) + xcr * cd.sin(X[2, k]))) / ((self.agent.lf + self.agent.lr) * cd.sqrt(
                    (X[0, k] - xcr) ** 2 + (X[1, k] - ycr) ** 2))

                b2 = Rl - cd.sqrt((X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2)

                lfb2 = params[10] * (Rl - cd.sqrt((X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2)) - (
                        X[3, k] * (
                        X[0, k] * cd.cos(X[2, k]) - xcl * cd.cos(X[2, k]) + X[1, k] * cd.sin(
                    X[2, k]) -
                        ycl * cd.sin(X[2, k]))) / cd.sqrt((X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2)

                l2fb2 = params[11] * (
                        params[10] * (Rl - cd.sqrt((X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2)) - (
                        X[3, k] * (
                        X[0, k] * cd.cos(X[2, k]) - xcl * cd.cos(X[2, k]) + X[1, k] * cd.sin(
                    X[2, k]) - ycl * cd.sin(X[2, k]))) / cd.sqrt(
                    (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2)) \
                        - X[3, k] * cd.cos(X[2, k]) * ((params[10] * (X[0, k] - xcl)) / cd.sqrt(
                    (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2) + (
                                                               X[3, k] * cd.cos(X[2, k])) / cd.sqrt(
                    (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2) - (X[3, k] * (2 * X[0, k] - 2 * xcl) * (
                        X[0, k] * cd.cos(X[2, k]) - xcl * cd.cos(X[2, k]) + X[1, k] * cd.sin(
                    X[2, k]) - ycl * cd.sin(X[2, k]))) / (
                                                               2 * ((X[0, k] - xcl) ** 2 + (
                                                               X[1, k] - ycl) ** 2) ** (3 / 2))) \
                        - X[3, k] * cd.sin(X[2, k]) * ((params[10] * (X[1, k] - ycl)) / cd.sqrt(
                    (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2) + (
                                                               X[3, k] * cd.sin(X[2, k])) / cd.sqrt(
                    (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2) - (X[3, k] * (2 * X[1, k] - 2 * ycl) * (
                        X[0, k] * cd.cos(X[2, k]) - xcl * cd.cos(X[2, k]) + X[1, k] * cd.sin(
                    X[2, k]) - ycl * cd.sin(X[2, k]))) / (
                                                               2 * ((X[0, k] - xcl) ** 2 + (
                                                               X[1, k] - ycl) ** 2) ** (3 / 2)))

                lgdelta2 = -(X[3, k] ** 2 * (
                        X[1, k] * cd.cos(X[2, k]) - ycl * cd.cos(X[2, k]) - X[0, k] * cd.sin(
                    X[2, k]) + xcl * cd.sin(X[2, k]))) / (
                                   (self.agent.lf + self.agent.lr) * cd.sqrt(
                               (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2))

                lgu2 = -(X[0, k] * cd.cos(X[2, k]) - xcl * cd.cos(X[2, k]) + X[1, k] * cd.sin(
                    X[2, k]) - ycl * cd.sin(X[2, k])) / cd.sqrt(
                    (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2)

                opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] + s[0, k] >= 0)
                opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[1, k] >= 0)

        if self.agent.road == 0:
            if self.mode == 1 or self.mode == 3:
                curr_states = self.mainroad_agent.get_states()
                for k in range(0, self.agent.N):
                    a = 1.5
                    b = 1
                    xip, yip, psi_ip, vip = curr_states
                    if self.cbf_type == 0:
                        b3 = (cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (X[1, k] - yip)) ** 2 / a + (
                                cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (X[0, k] - xip)) ** 2 / b - X[
                                 3, k] ** 2
                        lfb3 = 1 * (
                                (cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (X[1, k] - yip)) ** 2 / a +
                                (cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (X[0, k] - xip)) ** 2 / b -
                                X[3, k] ** 2) + X[3, k] * cd.cos(X[2, k]) * (
                                       (2 * cd.cos(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (
                                                   X[1, k] - yip))) / a +
                                       (2 * cd.sin(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (
                                                   X[0, k] - xip))) / b) - \
                               vip * cd.cos(psi_ip) * (
                                       (2 * cd.cos(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (
                                                   X[1, k] - yip))) / a +
                                       (2 * cd.sin(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (
                                                   X[0, k] - xip))) / b) + \
                               X[3, k] * cd.sin(X[2, k]) * ((2 * cd.cos(X[2, k]) * (
                                cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (X[0, k] - xip))) / b -
                                                            (2 * cd.sin(X[2, k]) * (
                                                                    cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(
                                                                X[2, k]) * (X[1, k] - yip))) / a) - \
                               vip * cd.sin(psi_ip) * (
                                       (2 * cd.cos(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (
                                                   X[0, k] - xip))) / b -
                                       (2 * cd.sin(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (
                                                   X[1, k] - yip))) / a)
                        lgb3delta = -(
                                X[3, k] * (
                                    2 * (cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (X[0, k] - xip)) *
                                    (cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (X[1, k] - yip)) / a -
                                    2 * (cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (X[0, k] - xip)) *
                                    (cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (
                                            X[1, k] - yip)) / b) / (
                                        self.agent.lf + self.agent.lr))
                        lgb3u = -2 * X[3, k]
                    else:
                        b3 = (X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2
                        lfb3 = (1 * ((X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2) +
                                (X[3, k] * cd.cos(X[2, k]) * (2 * X[0, k] - 2 * xip)) / a -
                                (vip * cd.cos(psi_ip) * (2 * X[0, k] - 2 * xip)) / a +
                                (X[3, k] * cd.sin(X[2, k]) * (2 * X[1, k] - 2 * yip)) / b -
                                (vip * cd.sin(psi_ip) * (2 * X[1, k] - 2 * yip)) / b)
                        lgb3u = -2 * X[3, k]
                        lgb3delta = 0

                    opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
                    curr_states = self.motion(self.dynamics, curr_states, [0, 0], [0, self.dt], self.dt)

            if self.mode == 2 or self.mode == 3:
                curr_states = self.mergingroad_agent.get_states()
                for k in range(0, self.agent.N):
                    xic, yic, psi_ic, vic = curr_states
                    R_i = cd.sqrt(x0 ** 2 + (y0 - 1000) ** 2)
                    R_c = cd.sqrt(xic ** 2 + (yic - 100) ** 2)
                    b01 = 1000
                    b00 = 100
                    delta = 0
                    theta_0_i = -1.4672603502370494
                    theta_0_ic = -0.51
                    const = 1.52 - 1.4672603502370494
                    b4 = (R_c * (theta_0_ic - np.arctan((yic - b00) / xic)) - R_i * (
                                theta_0_i - np.arctan((y0 - b01) / x0))
                          - (theta_0_i - np.arctan((y0 - b01) / x0)) / const * v0 - delta)

                    term1_numerator = X[3, k] * X[0, k] * cd.sin(X[2, k]) * (X[3, k] + R_i * const)
                    term1_denominator = const * (b01 ** 2 - 2 * b01 * X[1, k] + X[0, k] ** 2 + X[1, k] ** 2)
                    term1 = term1_numerator / term1_denominator

                    term2 = - (R_c * vic * cd.cos(psi_ic) * (b00 - yic)) / (
                            b00 ** 2 - 2 * b00 * yic + xic ** 2 + yic ** 2)
                    term3 = - (R_c * vic * xic * cd.sin(psi_ic)) / (b00 ** 2 - 2 * b00 * yic + xic ** 2 + yic ** 2)

                    term4 = params[12] * (delta + R_i * (theta_0_i + cd.atan2(b01 - X[1, k], X[0, k])) - R_c * (
                            theta_0_ic + cd.atan2(b00 - yic, xic)) + (
                                                  X[3, k] * (theta_0_i + cd.atan2(b01 - X[1, k], X[0, k]))) / const)

                    term5_numerator = X[3, k] * cd.cos(X[2, k]) * (b01 - X[1, k]) * (X[3, k] + R_i * const)
                    term5_denominator = const * (b01 ** 2 - 2 * b01 * X[1, k] + X[0, k] ** 2 + X[1, k] ** 2)
                    term5 = term5_numerator / term5_denominator
                    lfb4 = term1 + term2 + term3 - term4 + term5
                    lgb4u = -(theta_0_i + cd.atan2(b01 - X[1, k], X[0, k])) / const
                    lgb4delta = 0

                    opti.subject_to(lfb4 + lgb4u * u[0, k] + lgb4delta * u[1, k] + s[3, k] >= 0)
                    curr_states = self.motion(self.dynamics, curr_states, [0, 0], [0, self.dt], self.dt)

        else:
            if self.mode == 1 or self.mode == 3:
                curr_states = self.mergingroad_agent.get_states()
                for k in range(0, self.agent.N):
                    a = 1.5
                    b = 1
                    xip, yip, psi_ip, vip = curr_states
                    if self.cbf_type == 0:
                        b3 = (cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (X[1, k] - yip)) ** 2 / a + (
                                cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (X[0, k] - xip)) ** 2 / b - X[
                                 3, k] ** 2
                        lfb3 = 1 * (
                                (cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (X[1, k] - yip)) ** 2 / a +
                                (cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (X[0, k] - xip)) ** 2 / b -
                                X[3, k] ** 2) + X[3, k] * cd.cos(X[2, k]) * (
                                       (2 * cd.cos(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (
                                                   X[1, k] - yip))) / a +
                                       (2 * cd.sin(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (
                                                   X[0, k] - xip))) / b) - \
                               vip * cd.cos(psi_ip) * (
                                       (2 * cd.cos(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (
                                                   X[1, k] - yip))) / a +
                                       (2 * cd.sin(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (
                                                   X[0, k] - xip))) / b) + \
                               X[3, k] * cd.sin(X[2, k]) * ((2 * cd.cos(X[2, k]) * (
                                cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (X[0, k] - xip))) / b -
                                                            (2 * cd.sin(X[2, k]) * (
                                                                    cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(
                                                                X[2, k]) * (X[1, k] - yip))) / a) - \
                               vip * cd.sin(psi_ip) * (
                                       (2 * cd.cos(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (
                                                   X[0, k] - xip))) / b -
                                       (2 * cd.sin(X[2, k]) * (
                                                   cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (
                                                   X[1, k] - yip))) / a)
                        lgb3u = -2 * X[3, k]
                        lgb3delta = -(X[3, k] * (
                                2 * (cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (X[0, k] - xip)) *
                                (cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (X[1, k] - yip)) / a -
                                2 * (cd.cos(X[2, k]) * (X[1, k] - yip) + cd.sin(X[2, k]) * (X[0, k] - xip)) *
                                (cd.cos(X[2, k]) * (X[0, k] - xip) - cd.sin(X[2, k]) * (
                                        X[1, k] - yip)) / b) / (
                                              self.agent.lf + self.agent.lr))
                    else:
                        b3 = (X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2
                        lfb3 = (1 * ((X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2) +
                                (X[3, k] * cd.cos(X[2, k]) * (2 * X[0, k] - 2 * xip)) / a -
                                (vip * cd.cos(psi_ip) * (2 * X[0, k] - 2 * xip)) / a +
                                (X[3, k] * cd.sin(X[2, k]) * (2 * X[1, k] - 2 * yip)) / b -
                                (vip * cd.sin(psi_ip) * (2 * X[1, k] - 2 * yip)) / b)
                        lgb3u = -2 * X[3, k]
                        lgb3delta = 0

                    opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
                    curr_states = self.motion(self.dynamics, curr_states, [0, 0], [0, self.dt], self.dt)

            if self.mode == 2 or self.mode == 3:
                curr_states = self.mainroad_agent.get_states()
                for k in range(0, self.agent.N):
                    xic, yic, psi_ic, vic = curr_states
                    R_i = cd.sqrt(x0 ** 2 + (y0 - 100) ** 2)
                    R_c = cd.sqrt(x0 ** 2 + (y0 - 1000) ** 2)
                    b01 = 100
                    b00 = 1000
                    delta = 0
                    theta_0_i = -0.51
                    theta_0_ic = -1.4672603502370494
                    const = 1.52 - 0.51
                    b4 = (R_c * (theta_0_ic - np.arctan((yic - b00) / xic)) - R_i * (
                            theta_0_i - np.arctan((y0 - b01) / x0))
                          - (theta_0_i - np.arctan((y0 - b01) / x0)) / const * v0 - delta)

                    term1_numerator = X[3, k] * X[0, k] * cd.sin(X[2, k]) * (X[3, k] + R_i * const)
                    term1_denominator = const * (b01 ** 2 - 2 * b01 * X[1, k] + X[0, k] ** 2 + X[1, k] ** 2)
                    term1 = term1_numerator / term1_denominator

                    term2 = - (R_c * vic * cd.cos(psi_ic) * (b00 - yic)) / (
                            b00 ** 2 - 2 * b00 * yic + xic ** 2 + yic ** 2)
                    term3 = - (R_c * vic * xic * cd.sin(psi_ic)) / (b00 ** 2 - 2 * b00 * yic + xic ** 2 + yic ** 2)

                    term4 = params[13] * (delta + R_i * (theta_0_i + cd.atan2(b01 - X[1, k], X[0, k])) - R_c * (
                            theta_0_ic + cd.atan2(b00 - yic, xic)) + (
                                           X[3, k] * (theta_0_i + cd.atan2(b01 - X[1, k], X[0, k]))) / const)

                    term5_numerator = X[3, k] * cd.cos(X[2, k]) * (b01 - X[1, k]) * (X[3, k] + R_i * const)
                    term5_denominator = const * (b01 ** 2 - 2 * b01 * X[1, k] + X[0, k] ** 2 + X[1, k] ** 2)
                    term5 = term5_numerator / term5_denominator
                    lfb4 = term1 + term2 + term3 - term4 + term5
                    lgb4u = -(theta_0_i + cd.atan2(b01 - X[1, k], X[0, k])) / const
                    lgb4delta = 0

                    opti.subject_to(lfb4 + lgb4u * u[0, k] + lgb4delta * u[1, k] + s[3, k] >= 0)
                    curr_states = self.motion(self.dynamics, curr_states, [0, 0], [0, self.dt], self.dt)

        for k in range(0, self.agent.N):
            b = X[3, k] - self.agent.vmin
            lfb = params[14] * b
            lfb = 1 * b
            lgu = 1
            opti.subject_to(lfb + lgu * u[0, k] >= 0)

            b = -X[3, k] + self.agent.vmax
            lfb = params[15] * b
            lfb = 1 * b
            lgu = -1
            opti.subject_to(lfb + lgu * u[0, k] >= 0)

        # Define the cost function
        cost = 0
        diag_elements_u = [params[16], params[17],params[18], params[19]]
        u_ref = np.zeros((self.agent.Inputnumber + 2, self.agent.N))

        normalization_factor = [max(-self.agent.umin,self.agent.umax),max(self.agent.steermax,-self.agent.steermin),200,0.4]
        for i in range(self.agent.Inputnumber + 2):
            for h in range(self.agent.N):
                cost += 0.5 * diag_elements_u[i] * ((u[i, h] - u_ref[i][h])/normalization_factor[i]) ** 2
                cost += 10 ** 8 * (s[0, h]) ** 2
                cost += 10 ** 8 * (s[1, h]) ** 2
                cost += 10 ** 8 * (s[2, h]) ** 2
                cost += 10 ** 8 * (s[3, h]) ** 2

        opti.subject_to(self.agent.umin <= u[0, :])
        opti.subject_to(u[0, :] <= self.agent.umax)
        eps3 = 1
        for k in range(0, self.agent.N):
            V = (X[3, k] - 15) ** 2
            lfV = eps3 * V
            lgu = 2 * (X[3, k] - 15)
            opti.subject_to(lfV + lgu * u[0, k] - u[2, k] <= 0)

            V = (X[2, k] - Psi_ref_path_continous) ** 2
            lfV = 10*eps3 * V
            lgdelta = 2 * (X[2, k] - Psi_ref_path_continous) * X[3, k] / (self.agent.lr + self.agent.lf)
            opti.subject_to(lfV + lgdelta * u[1, k] - u[3, k] <= 0)

        opti.subject_to(self.agent.steermin <= u[1, :])
        opti.subject_to(u[1, :] <= self.agent.steermax)

        opti.subject_to(X[:, 0] == x_init)  # initialize states
        timespan = [0, self.dt]

        for h in range(self.agent.N):  # initial guess
            opti.set_initial(X[:, h], [Ref_Centerlane[h][0], Ref_Centerlane[h][1], Psi_ref[h], v0])

        for k in range(self.agent.N):
            state = []
            Input = []
            for j in range(self.agent.statesnumber):
                state.append(X[j, k])
            for j in range(0, self.agent.Inputnumber):
                Input.append(u[j, k])
            state = self.agent.rk4(timespan, state, Input, 1)
            for j in range(self.agent.statesnumber):
                opti.subject_to(X[j, k + 1] == state[j])

        try:
            opts = {}
            opts['print_time'] = False
            opts['ipopt.print_level'] = False
            opti.solver('ipopt', opts)
            opti.minimize(cost)
            sol = opti.solve()

            if self.agent.N > 1:
                self.agent.s_vars = sol.value(s)[:, 0]
                if np.any(self.agent.s_vars > 0.1):
                    self.agent.s_vars = [min(sol.value(s)[i, 0], 0.1) for i in range(len(sol.value(s)[:, 0]))]
                    return "No solution found", np.array([-5.66, 0])
                else:
                    return "solution found", sol.value(u)[:2, 0]
            else:
                self.agent.s_vars = sol.value(s)
                if np.any(sol.value(s) > 0.1):
                    self.agent.s_vars = [0.1] * 4
                    return "No solution found", np.array([-5.66, 0])
                else:
                    return "solution found", sol.value(u[:2])

        except:
            return "No solution found", np.array([-5.66, 0])
    def control_other_agent(self, x_init, env_agent):

        if x_init[0] >= 0:
            params = [0.9] * 18
            env_agent.referencegenerator()
            Ref_Rightlane = env_agent.Ref_Rightlane
            Ref_Leftlane = env_agent.Ref_Leftlane
            Ref_Centerlane = env_agent.Ref_Centerlane
            Psi_ref = env_agent.Ref_Psi

            x0 = x_init[0]
            y0 = x_init[1]
            psi0 = x_init[2]
            v0 = x_init[3]

            G = np.array([[1, 0, 0, 0], [-1, 0, 0, 0]])
            h = np.array([env_agent.umax, -env_agent.umin])

            u = cp.Variable(4)

            Ref_Rightlane_x = [Ref_Rightlane[i][0] for i in range(0, len(Ref_Rightlane))]
            Ref_Rightlane_y = [Ref_Rightlane[i][1] for i in range(0, len(Ref_Rightlane))]
            Ref_Leftlane_x = [Ref_Leftlane[i][0] for i in range(0, len(Ref_Leftlane))]
            Ref_Leftlane_y = [Ref_Leftlane[i][1] for i in range(0, len(Ref_Leftlane))]

            z_r = np.polyfit(Ref_Rightlane_x, Ref_Rightlane_y, 1)
            z_l = np.polyfit(Ref_Leftlane_x, Ref_Leftlane_y, 1)
            env_agent.perceivedlane_r = z_r
            env_agent.perceivedlane_l = z_l

            if len(Ref_Rightlane) < 3:
                while (len(Ref_Rightlane) < 3):
                    index = len(Ref_Rightlane) - 1
                    newpoint_x = 0.5 * (Ref_Rightlane[index][0] + self.agent.coordinates.X_R[-2 - index])
                    newpoint_y = 0.5 * (Ref_Rightlane[index][1] + self.agent.coordinates.Y_R[-2 - index])
                    Ref_Rightlane.append([newpoint_x, newpoint_y])
                    newpoint_x = 0.5 * (Ref_Leftlane[index][0] + self.agent.coordinates.X_L[-2 - index])
                    newpoint_y = 0.5 * (Ref_Leftlane[index][1] + self.agent.coordinates.Y_L[-2 - index])
                    Ref_Leftlane.append([newpoint_x, newpoint_y])

            xcr, ycr, Rr, sigma = taubinSVD(Ref_Rightlane)
            xcl, ycl, Rl, sigma = taubinSVD(Ref_Leftlane)

            if Rr >= 1200 or Rl >= 1200:
                if abs(np.sin(Psi_ref[0])) <= 0.05:
                    if y0 - Ref_Rightlane[0][1] >= 0:
                        b1 = y0 - Ref_Rightlane[0][1]
                        lfb1 = v0 * np.sin(psi0) + params[0] * b1
                        l2fb1 = params[0] * (lfb1 - params[0] * b1) + params[1] * lfb1
                        lgu1 = np.sin(psi0)
                        lgdelta1 = v0 ** 2 * np.cos(psi0) / (env_agent.lf + env_agent.lr)
                        G = np.append(G, [[-lgu1, -lgdelta1, 0, 0]], axis=0)
                        h = np.append(h, [l2fb1])

                        b2 = -y0 + Ref_Leftlane[0][1]
                        lfb2 = -v0 * np.sin(psi0) + params[2] * b2
                        l2fb2 = params[2] * (lfb2 - params[2] * b2) + params[3] * lfb2
                        lgu2 = -np.sin(psi0)
                        lgdelta2 = -v0 ** 2 * np.cos(psi0) / (env_agent.lf + env_agent.lr)
                        G = np.append(G, [[-lgu2, -lgdelta2, 0, 0]], axis=0)
                        h = np.append(h, [l2fb2])
                    else:
                        b1 = y0 - Ref_Leftlane[0][1]
                        lfb1 = v0 * np.sin(psi0) + params[4] * b1
                        l2fb1 = params[4] * (lfb1 - params[4] * b1) + params[5] * lfb1
                        lgu1 = np.sin(psi0)
                        lgdelta1 = v0 * np.cos(psi0) * (v0) / (env_agent.lf + env_agent.lr)
                        G = np.append(G, [[-lgu1, -lgdelta1, 0, 0]], axis=0)
                        h = np.append(h, [l2fb1])

                        b2 = -y0 + Ref_Rightlane[0][1]
                        lfb2 = -v0 * np.sin(psi0) + params[6] * b2
                        l2fb2 = params[6] * (lfb2 - params[6] * b2) + params[7] * lfb2
                        lgu2 = -np.sin(psi0)
                        lgdelta2 = -v0 * np.cos(psi0) * (v0) / (env_agent.lf + env_agent.lr)
                        G = np.append(G, [[-lgu2, -lgdelta2, 0, 0]], axis=0)
                        h = np.append(h, [l2fb2])

            else:
                b1 = np.sqrt((x0 - xcr) ** 2 + (y0 - ycr) ** 2) - Rr
                lfb1 = (v0 * (x0 * np.cos(psi0) - xcr * np.cos(psi0) + y0 * np.sin(psi0) - ycr * np.sin(
                    psi0))) / np.sqrt(
                    (x0 - xcr) ** 2 + (y0 - ycr) ** 2) - \
                    params[8] * (Rr - np.sqrt((x0 - xcr) ** 2 + (y0 - ycr) ** 2))

                l2fb1 = v0 * np.cos(psi0) * (
                        (params[8] * (x0 - xcr)) / np.sqrt((x0 - xcr) ** 2 + (y0 - ycr) ** 2) + (
                        v0 * np.cos(psi0)) / np.sqrt((x0 - xcr) ** 2 +
                                                    (y0 - ycr) ** 2) - (v0 * (2 * x0 - 2 * xcr) * (
                        x0 * np.cos(psi0) - xcr * np.cos(psi0) + y0 * np.sin(psi0) - ycr * np.sin(psi0))) / (
                                2 * ((x0 - xcr) ** 2 +
                                    (y0 - ycr) ** 2) ** (3 / 2))) - params[9] * (
                                params[8] * (Rr - np.sqrt((x0 - xcr) ** 2 + (y0 - ycr) ** 2)) - (
                                v0 * (x0 * np.cos(psi0) - xcr * np.cos(psi0) +
                                    y0 * np.sin(psi0) - ycr * np.sin(psi0))) / np.sqrt(
                            (x0 - xcr) ** 2 + (y0 - ycr) ** 2)) + v0 * np.sin(psi0) * (
                                (params[8] * (y0 - ycr)) / np.sqrt((x0 - xcr) ** 2 +
                                                                (y0 - ycr) ** 2) + (v0 * np.sin(psi0)) / np.sqrt(
                            (x0 - xcr) ** 2 + (y0 - ycr) ** 2) - (v0 * (2 * y0 - 2 * ycr) * (x0 * np.cos(psi0) -
                                                                                            xcr * np.cos(
                                    psi0) + y0 * np.sin(psi0) - ycr * np.sin(psi0))) / (
                                        2 * ((x0 - xcr) ** 2 + (y0 - ycr) ** 2) ** (3 / 2)))

                lgu1 = (x0 * np.cos(psi0) - xcr * np.cos(psi0) + y0 * np.sin(psi0) - ycr * np.sin(psi0)) / np.sqrt(
                    (x0 - xcr) ** 2 + (y0 - ycr) ** 2)

                lgdelta1 = (v0 ** 2 * (
                        y0 * np.cos(psi0) - ycr * np.cos(psi0) - x0 * np.sin(psi0) + xcr * np.sin(psi0))) / (
                                (env_agent.lf + env_agent.lr) * np.sqrt((x0 - xcr) ** 2 + (y0 - ycr) ** 2))

                b2 = Rl - np.sqrt((x0 - xcl) ** 2 + (y0 - ycl) ** 2)
                lfb2 = params[10] * (Rl - np.sqrt((x0 - xcl) ** 2 + (y0 - ycl) ** 2)) - (
                        v0 * (x0 * np.cos(psi0) - xcl * np.cos(psi0) + y0 * np.sin(psi0) -
                            ycl * np.sin(psi0))) / np.sqrt((x0 - xcl) ** 2 + (y0 - ycl) ** 2)

                l2fb2 = params[11] * (params[10] * (Rl - np.sqrt((x0 - xcl) ** 2 + (y0 - ycl) ** 2)) - (
                        v0 * (x0 * np.cos(psi0) - xcl * np.cos(psi0) + y0 * np.sin(psi0) - ycl * np.sin(
                    psi0))) / np.sqrt(
                    (x0 - xcl) ** 2 + (y0 - ycl) ** 2)) \
                        - v0 * np.cos(psi0) * (
                                (params[10] * (x0 - xcl)) / np.sqrt((x0 - xcl) ** 2 + (y0 - ycl) ** 2) + (
                                v0 * np.cos(psi0)) / np.sqrt((x0 - xcl) ** 2 + (y0 - ycl) ** 2) \
                                - (v0 * (2 * x0 - 2 * xcl) * (
                                x0 * np.cos(psi0) - xcl * np.cos(psi0) + y0 * np.sin(psi0) - ycl * np.sin(
                            psi0))) / (
                                        2 * ((x0 - xcl) ** 2 + (y0 - ycl) ** 2) ** (3 / 2))) \
                        - v0 * np.sin(psi0) * (
                                (params[10] * (y0 - ycl)) / np.sqrt((x0 - xcl) ** 2 + (y0 - ycl) ** 2) + (
                                v0 * np.sin(psi0)) / np.sqrt((x0 - xcl) ** 2 + (y0 - ycl) ** 2) \
                                - (v0 * (2 * y0 - 2 * ycl) * (
                                x0 * np.cos(psi0) - xcl * np.cos(psi0) + y0 * np.sin(psi0) - ycl * np.sin(
                            psi0))) / (
                                        2 * ((x0 - xcl) ** 2 + (y0 - ycl) ** 2) ** (3 / 2)))

                lgdelta2 = -(v0 ** 2 * (
                        y0 * np.cos(psi0) - ycl * np.cos(psi0) - x0 * np.sin(psi0) + xcl * np.sin(psi0))) / (
                                (env_agent.lf + env_agent.lr) * np.sqrt((x0 - xcl) ** 2 + (y0 - ycl) ** 2))

                lgu2 = -(x0 * np.cos(psi0) - xcl * np.cos(psi0) + y0 * np.sin(psi0) - ycl * np.sin(psi0)) / np.sqrt(
                    (x0 - xcl) ** 2 + (y0 - ycl) ** 2)

                G = np.append(G, [[-lgu1, -lgdelta1, 0, 0]], axis=0)
                h = np.append(h, [l2fb1])
                G = np.append(G, [[-lgu2, -lgdelta2, 0, 0]], axis=0)
                h = np.append(h, [l2fb2])

            Ref_Centerlane_x = [Ref_Centerlane[i][0] for i in range(0, len(Ref_Centerlane))]
            Ref_Centerlane_y = [Ref_Centerlane[i][1] for i in range(0, len(Ref_Centerlane))]
            waypoints2centerx = [x0] + Ref_Centerlane_x[2:8]
            waypointstraj2centery = [y0] + Ref_Centerlane_y[2:8]
            poly = np.polyfit(waypoints2centerx, waypointstraj2centery, 2)

            x1 = np.linspace(waypoints2centerx[0], waypoints2centerx[-1], len(Psi_ref))
            y1 = np.linspace(waypointstraj2centery[0], waypointstraj2centery[-1], len(Psi_ref))
            Psi_ref_path = [np.arctan2((y1[i + 1] - y1[i]), (x1[i + 1] - x1[i])) for i in range(len(Psi_ref) - 1)]
            Psi_ref_path_continous = [(psi_ref-self.agent.last_psi_ref_path + np.pi)%(2*np.pi)-np.pi + self.agent.last_psi_ref_path for psi_ref in Psi_ref_path]
            self.agent.last_psi_ref_path = Psi_ref[0]
            G = np.append(G, [[0, 1, 0, 0]], axis=0)
            h = np.append(h, [env_agent.steermax])

            G = np.append(G, [[0, -1, 0, 0]], axis=0)
            h = np.append(h, [-env_agent.steermin])

            b = v0 - env_agent.vmin
            lfb = params[12] * b
            lgu = 1
            G = np.append(G, [[-lgu, 0, 0, 0]], axis=0)
            h = np.append(h, [lfb])

            b = -v0 + env_agent.vmax
            lfb = params[13] * b
            lgu = -1
            G = np.append(G, [[-lgu, 0, 0, 0]], axis=0)
            h = np.append(h, [lfb])

            eps3 = 1

            V = (v0 - env_agent.desired_speed) ** 2
            lfV = eps3 * V
            lgu = 2 * (v0 - env_agent.desired_speed)
            lgdelta = 0
            G = np.append(G, [[lgu, lgdelta, -1, 0]], axis=0)
            h = np.append(h, [-lfV])

            eps3 = 10
            V = (psi0 - Psi_ref_path_continous[0]) ** 2
            lfV = eps3 * V
            lgu = 0
            lgdelta = 2 * (psi0 - Psi_ref_path_continous[0]) * (v0) / (env_agent.lf + env_agent.lr)
            G = np.append(G, [[lgu, lgdelta, 0, -1]], axis=0)
            h = np.append(h, [-lfV])

            A = [0, 0, 0, 0]
            b = [0]
            P = np.array(
                [[params[14], 0, 0, 0], [0, params[15], 0, 0], [0, 0, params[16], 0], [0, 0, 0, params[17]]])
            q = np.array([0, 0, 0, 0])
            prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(u, P) + q.T @ u),
                            [G @ u <= h,
                            A @ u == b])
            try:
                prob.solve()
                if self.evalmode == 0:
                    acc_randomness = random.uniform(-5, 3)
                    steer_randomness = 0.02 * random.uniform(-1, 1)
                else:
                    acc_randomness =  0
                    steer_randomness = 0

                return "solution found", np.array([[np.clip(u.value[0]+acc_randomness, self.agent.umin,self.agent.umax)] , [u.value[1]+steer_randomness]])
            except:
                return "No solution found", np.array([[-5.66], [0]])
            
        else:

            return "No control", np.array([[+5.66], [0]])

    def render(self):
        if self.render_mode == "Visualization":
            self.fig.suptitle('vehicle path')
            self.ax.plot(self.coordinates.mainroad['X_C'],
                         self.coordinates.mainroad['Y_C'], 'y--')
            self.ax.plot(self.coordinates.mainroad['X_R'][:-97 - 120],
                         self.coordinates.mainroad['Y_R'][:-97 - 120], 'k')
            self.ax.plot(self.coordinates.mainroad['X_L'],
                         self.coordinates.mainroad['Y_L'], 'k')
            self.ax.plot(self.coordinates.mergingroad['X_C'],
                         self.coordinates.mergingroad['Y_C'], 'y--')
            self.ax.plot(self.coordinates.mergingroad['X_L'][:-97 - 120],
                         self.coordinates.mergingroad['Y_L'][:-97 - 120], 'k')
            self.ax.plot(self.coordinates.mergingroad['X_R'],
                         self.coordinates.mergingroad['Y_R'], 'k')

            Drawing_colored_circle0 = plt.Circle((self._agent_states[0], self._agent_states[1]), 1.5)
            self.ax.set_aspect(1)
            self.ax.add_artist(Drawing_colored_circle0)
            if self.agent.road == 0:
                if self.mode == 1 or self.mode == 3:
                    Drawing_colored_circle1 = plt.Circle((self.mainroad_agent.x, self.mainroad_agent.y), 1.5)
                    self.ax.set_aspect(1)
                    self.ax.add_artist(Drawing_colored_circle1)
                if self.mode == 2 or self.mode == 3:
                    Drawing_colored_circle2 = plt.Circle((self.mergingroad_agent.x, self.mergingroad_agent.y), 1.5)
                    self.ax.set_aspect(1)
                    self.ax.add_artist(Drawing_colored_circle2)
            else:
                if self.mode == 1 or self.mode == 3:
                    Drawing_colored_circle2 = plt.Circle((self.mergingroad_agent.x, self.mergingroad_agent.y), 1.5)
                    self.ax.set_aspect(1)
                    self.ax.add_artist(Drawing_colored_circle2)
                if self.mode == 2 or self.mode == 3:
                    Drawing_colored_circle1 = plt.Circle((self.mainroad_agent.x, self.mainroad_agent.y), 1.5)
                    self.ax.set_aspect(1)
                    self.ax.add_artist(Drawing_colored_circle1)

            self.fig.canvas.draw()
            time.sleep(0.1)
            self.fig.canvas.flush_events()
            self.ax.clear()
            return None





