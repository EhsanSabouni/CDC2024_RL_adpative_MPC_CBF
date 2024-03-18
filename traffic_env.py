import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

class traffic_env():
    def __init__(self, total_num_cars, dt, N, render_mode):
        self.map_pointer = 0
        self.car_pointer = 0
        self.simindex = 0
        self.N = N
        self.timeout = 30
        self.total_num_cars = total_num_cars
        metrics_list = ['time', 'pos_x', 'pos_y', 'psi', 'speed', 'acc', 'steer', 'energy', 'time', 'fuel']
        self.metrics = [0] * self.total_num_cars
        for index in range(0, self.total_num_cars):
            self.metrics[index] = {key: [] for key in metrics_list}
        self.dt = dt
        self.ave_energy = 0
        self.ave_time = 0
        self.ave_fuel = 0
        self.num_cars_leave = 0
        self.simtime = []
        # self.metrics = {}
        self.mainroad = {'X_R':[],  # X_R is the x coordinate of the right boundary
                         'Y_R':[],  # Y_R is the y coordinate of the right boundary
                         'X_L':[],  # X_L is the x coordinate of the left boundary
                         'Y_L':[],  # Y_L is the y coordinate of the left boundary
                         'X_C':[],  # X_C is the x coordinate of the centerline
                         'Y_C':[], # Y_C is the y coordinate of the centerline
                         'Psi_Int':[]} # psiInt is the integral of the psi angle

        self.mergingroad = {'X_R':[],
                         'Y_R':[],
                         'X_L':[],
                         'Y_L':[],
                         'X_C':[], # X_C is the x coordinate of the centerline
                         'Y_C': [], # Y_C is the y coordinate of the centerline
                         'Psi_Int':[]}  # psiInt is the integral of the psi angle
        self.render_mode = render_mode
        if render_mode == 'Visualization':
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8])
    def mapcoordinates(self):
        for key in self.mainroad:
            # self.mainroad[key] = pd.read_csv("G:\My Drive\cav-control-mpc_\RoadsData\MainRoad_data - Copy.csv")[key].to_numpy()[self.map_pointer:]
            # self.mergingroad[key] = pd.read_csv("G:\My Drive\cav-control-mpc_\RoadsData\MergingRoad_data - Copy.csv")[key].to_numpy()[self.map_pointer:]
            self.mainroad[key] = pd.read_csv("~/Desktop/GitHub/Merging/multi-agent-qp/RoadData/MainRoad_data_merging1.csv")[
                                     key].to_numpy()[self.map_pointer:]
            self.mergingroad[key] = pd.read_csv("~/Desktop/GitHub/Merging/multi-agent-qp/RoadData/MergingRoad_data_merging1.csv")[
                                        key].to_numpy()[self.map_pointer:]

        return None

    def step(self, cars):
        for car in cars:
            timespan = [0, self.dt]
            'Moving the vehicle'
            CurrState = [car.x, car.y, car.psi, car.v]
            action = [car.acc, car.steering]
            car.x, car.y, car.psi, car.v = car.motion(car.dynamics, CurrState, action, timespan, self.dt)
            car.metric_update(self.dt)

        self.metric_update(cars, 0, 0)


        return None

    def reset(self):
        self.ave_energy = 0
        self.ave_time = 0
        self.ave_fuel = 0
        self.num_cars_leave = 0
        self.car_pointer = 0
        self.simindex = 0
        metrics_list = ['time', 'pos_x', 'pos_y', 'psi', 'speed', 'acc', 'steer', 'energy', 'time', 'fuel']
        self.metrics = [0] * self.total_num_cars
        for index in range(0, self.total_num_cars):
            self.metrics[index] = {key: [] for key in metrics_list}
    def checkconflicts(self, ego, que, index):

        ip, ic = None, None
        for i in range(0, index):
            if ego.road == que[i].road:
                ip = i
            else:
                ic = i

        if ip == None:
            ego.ip_states = [-1000, 0, 0, 0]
        else:
            ego.ip_states = [que[ip].x, que[ip].y, que[ip].psi, que[ip].v]

        if ic == None:
            ego.ic_states = [-1000, 0, 0, 0]
        else:
            ego.ic_states = [que[ic].x, que[ic].y, que[ic].psi, que[ic].v]

        return ego

    def checkleave(self, ego, index, ind):
        end_pointer = -55
        thr = 30
        if ego.road == 0:
            end_coord = [ego.coordinates.mainroad['X_C'][end_pointer], ego.coordinates.mainroad['Y_C'][end_pointer]]
        else:
            end_coord = [ego.coordinates.mergingroad['X_C'][end_pointer],
                         ego.coordinates.mergingroad['Y_C'][end_pointer]]

        Dist2MP = (ego.x - end_coord[0]) ** 2 + (ego.y - end_coord[1]) ** 2

        if Dist2MP < thr:
            return ind + [index]

        return ind
    def metric_update(self, cars, index, flag):
        if flag == 1:
            for ind in index:
                self.ave_energy = (self.ave_energy * self.num_cars_leave + cars[ind].energy)/(self.num_cars_leave + 1)
                self.ave_fuel = (self.ave_fuel * self.num_cars_leave + cars[ind].fuel) / (self.num_cars_leave + 1)
                self.ave_time = (self.ave_time * self.num_cars_leave + cars[ind].time) / (self.num_cars_leave + 1)
                self.num_cars_leave += 1
        else:
            id_tuple = [(cars[k].id, k) for k in range(len(cars))]
            for id in id_tuple:
                self.metrics[id[0]]['time'].append(cars[id[1]].time)
                self.metrics[id[0]]['pos_x'].append(cars[id[1]].x)
                self.metrics[id[0]]['pos_y'].append(cars[id[1]].y)
                self.metrics[id[0]]['psi'].append(cars[id[1]].psi)
                self.metrics[id[0]]['speed'].append(cars[id[1]].v)
                self.metrics[id[0]]['acc'].append(cars[id[1]].acc)
                self.metrics[id[0]]['steer'].append(cars[id[1]].steering)
                self.metrics[id[0]]['energy'].append(cars[id[1]].energy)
                self.metrics[id[0]]['fuel'].append(cars[id[1]].fuel)



    def render(self, vehicles):
        if self.render_mode == 'Visualization':
            # if self.traj == 'Merging lane':
            #     ax.set_xlim(left=-200, right=0)
            #     ax.set_ylim(bottom=-400, top=-200)
            #     # img = plt.imread("Mass_turnpike.png")
            #     # ax.imshow(img, extent=[-405, -170, -310, -225])
            # elif self.traj == 'Main lane':
            #     ax.set_xlim(left=-140, right=40)
            #     ax.set_ylim(bottom=-320, top=-140)
            # elif self.traj == 'MergingRoadways':
            #     ax.set_xlim(left=-170, right=50)
            #     ax.set_ylim(bottom=-325, top=-150)


            self.fig.suptitle('vehicle path')
            self.ax.plot(self.mainroad['X_C'], self.mainroad['Y_C'], 'y--')
            self.ax.plot(self.mainroad['X_R'][:-97-120], self.mainroad['Y_R'][:-97-120], 'k')
            self.ax.plot(self.mainroad['X_L'], self.mainroad['Y_L'], 'k')
            self.ax.plot(self.mergingroad['X_C'], self.mergingroad['Y_C'], 'y--')
            self.ax.plot(self.mergingroad['X_L'][:-97-120], self.mergingroad['Y_L'][:-97-120], 'k')
            self.ax.plot(self.mergingroad['X_R'], self.mergingroad['Y_R'], 'k')

            # fig.suptitle('vehicle path')
            # ax.plot(self.mainroad['X_C'], self.mainroad['Y_C'], 'y--')
            # ax.plot(self.mainroad['X_L'], self.mainroad['Y_L'], 'k')
            # ax.plot(self.mainroad['X_R'][: 197 - self.map_pointer], self.mainroad['Y_R'][:197 - self.map_pointer], 'k')
            # ax.plot(self.mergingroad['X_C'], self.mergingroad['Y_C'], 'y--')
            # ax.plot(self.mergingroad['X_L'], self.mergingroad['Y_L'], 'k')
            # ax.plot(self.mergingroad['X_R'][:223 - self.map_pointer], self.mergingroad['Y_R'][:223 - self.map_pointer], 'k')
            # ax.plot(self.mainroad['X_C'], self.mainroad['Y_C'], 'y--')
            # ax.plot(self.mainroad['X_L'], self.mainroad['Y_L'], 'k')
            # ax.plot(self.mainroad['X_R'], self.mainroad['Y_R'], 'k')
            # ax.plot(self.mergingroad['X_C'], self.mergingroad['Y_C'], 'y--')
            # ax.plot(self.mergingroad['X_L'], self.mergingroad['Y_L'], 'k')
            # ax.plot(self.mergingroad['X_R'], self.mergingroad['Y_R'], 'k')
            # img = plt.imread("airlines.jpg")
            # fig, ax = plt.subplots()
            # ax.imshow(img)
            for car in range(0, len(vehicles)):

                Ref_Centerlane = vehicles[car].Ref_Centerlane

                # if with_prediction == 1:
                #     ax.plot(vehicles[car].predictedstate[:, 0], vehicles[car].predictedstate[:, 1], 'r')
                Drawing_colored_circle = plt.Circle((vehicles[car].x, vehicles[car].y), 1.5)
                # delta = 0.1
                # u = vehicles[car].x  # x-position of the center
                # v = vehicles[car].y  # y-position of the center
                # a = vehicles[car].p[0] * vehicles[car].v + delta  # radius on the x-axis
                # b = vehicles[car].p[1] * vehicles[car].v + delta # radius on the y-axis
                # t_rot = vehicles[car].psi  # rotation angle
                #
                # t = np.linspace(0, 2 * np.pi, 100)
                # Ell = np.array([a * np.cos(t), b * np.sin(t)])
                # # u,v removed to keep the same center location
                # R_rot = np.array([[np.cos(t_rot), -np.sin(t_rot)], [np.sin(t_rot), np.cos(t_rot)]])
                # # 2-D rotation matrix
                # Ell_rot = np.zeros((2, Ell.shape[1]))
                # for i in range(Ell.shape[1]):
                #     Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
                #
                # # plt.plot( u+Ell[0,:] , v+Ell[1,:] )     #initial ellipse
                # plt.plot(u + Ell_rot[0, :], v + Ell_rot[1, :], 'darkorange')  # rotated ellipse
                # plt.grid(color='lightgray', linestyle='--')
                self.ax.set_aspect(1)
                self.ax.add_artist(Drawing_colored_circle)

                Ref_Centerlane_x = [Ref_Centerlane[i][0] for i in range(0, len(Ref_Centerlane))]
                Ref_Centerlane_y = [Ref_Centerlane[i][1] for i in range(0, len(Ref_Centerlane))]
                waypoints2centerx =[vehicles[car].x] + Ref_Centerlane_x[5:10]
                waypointstraj2centery =[vehicles[car].y] + Ref_Centerlane_y[5:10]
                # z = np.polyfit(waypoints2centerx, waypointstraj2centery, 2)
                # x_fit = np.linspace(vehicles[car].x, waypoints2centerx[-1], num=100)
                # x_fit = np.linspace(vehicles[car].x, waypoints2centerx[-1] , num=100)
                # y_fit = vehicles[car].poly[0] * x_fit ** 2 + vehicles[car].poly[1] * x_fit + vehicles[car].poly[2]
                # ax.plot(x_fit, y_fit, 'b')

                # Ref_Leftlane = vehicles[car].cameradata["Ref_Leftlane"]
                # Ref_Leftlane_x = [Ref_Leftlane[i][0] for i in range(0, len(Ref_Leftlane))]
                #
                # Ref_Rightlane = vehicles[car].cameradata["Ref_Rightlane"]
                # Ref_Rightlane_x = [Ref_Rightlane[i][0] for i in range(0, len(Ref_Rightlane))]
                #
                #
                # x_fit_l = np.linspace(Ref_Leftlane_x[0], Ref_Leftlane_x[-1], num=10)
                # x_fit_r = np.linspace(Ref_Rightlane_x[0], Ref_Rightlane_x[-1], num=10)
                # if vehicles[car].perceivedlane_ploytype == 1:
                #     y_fit_l = vehicles[car].perceivedlane_l[0] * x_fit_l + vehicles[car].perceivedlane_l[1]
                #     y_fit_r = vehicles[car].perceivedlane_r[0] * x_fit_r + vehicles[car].perceivedlane_r[1]
                #     ax.plot(x_fit_l, y_fit_l, 'b')
                #     ax.plot(x_fit_r, y_fit_r, 'b')
            # u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            # x = obstacle[3] * np.cos(u) * np.sin(v) + obstacle[0]
            # y = obstacle[3] * np.sin(u) * np.sin(v) + obstacle[1]
            # z = obstacle[3] * np.cos(v) + obstacle[2]
            # ax.plot_wireframe(x, y, z, color="red")
            self.fig.canvas.draw()
            # time.sleep(0.1)
            self.fig.canvas.flush_events()
            self.ax.clear()
        return None


