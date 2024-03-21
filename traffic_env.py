import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.image as mpimg
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import cv2
import warnings
warnings.filterwarnings("ignore")

class traffic_env():
    def __init__(self, total_num_cars, dt, N, render_mode):
        self.pointer = 0
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
            current_directory = os.path.dirname(__file__)
            main_road_path = os.path.join(current_directory, 'RoadData/MainRoad_data_merging1.csv')
            merging_road_path = os.path.join(current_directory, 'RoadData/MergingRoad_data_merging1.csv')
            self.mainroad[key] = pd.read_csv(main_road_path)[key].to_numpy()[self.pointer:]
            self.mergingroad[key] = pd.read_csv(merging_road_path)[key].to_numpy()[self.pointer:]

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

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))

    def render(self, vehicles):
        if self.render_mode == 'Visualization':
            self.fig.suptitle('vehicle path')
            self.ax.plot(self.mainroad['X_C'], self.mainroad['Y_C'], 'y--')
            self.ax.plot(self.mainroad['X_R'][:-97 - 120], self.mainroad['Y_R'][:-97 - 120], 'k')
            self.ax.plot(self.mainroad['X_L'], self.mainroad['Y_L'], 'k')
            self.ax.plot(self.mergingroad['X_C'], self.mergingroad['Y_C'], 'y--')
            self.ax.plot(self.mergingroad['X_L'][:-97 - 120], self.mergingroad['Y_L'][:-97 - 120], 'k')
            self.ax.plot(self.mergingroad['X_R'], self.mergingroad['Y_R'], 'k')
            for car in range(0, len(vehicles)):

                # Set initial position and orientation
                x = vehicles[car].x  # initial x-coordinate
                y = vehicles[car].y  # initial y-coordinate
                if vehicles[car].road == 0:
                    coeff = 5
                else:
                    if vehicles[car].x >= 2.5:
                        coeff = 15
                    else:
                        coeff = 5

                theta = 180 - coeff - (vehicles[car].psi / np.pi * 180)  # initial orientation angle (in radians)
                # Transformation matrix for translation and rotation
                current_directory = os.path.dirname(__file__)
                phot_dir = os.path.join(current_directory, 'car.png')
                img = mpimg.imread(phot_dir)
                rotated = self.rotate_bound(img, theta)
                rotated_image_clipped = np.clip(rotated, 0, 1)
                imagebox = OffsetImage(rotated_image_clipped, zoom=0.08)

                # Annotation box for solar pv logo
                ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                self.ax.add_artist(ab)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.ax.clear()
        return None



