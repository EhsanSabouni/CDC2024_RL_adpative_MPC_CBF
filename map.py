import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
class map:
    def __init__(self, pointer: int):
        self.pointer = pointer
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

    def mapcoordinates(self):
        for key in self.mainroad:
            current_directory = os.path.dirname(__file__)
            main_road_path = os.path.join(current_directory, 'RoadData/MainRoad_data_merging1.csv')
            merging_road_path = os.path.join(current_directory, 'RoadData/MergingRoad_data_merging1.csv')
            self.mainroad[key] = pd.read_csv(main_road_path)[key].to_numpy()[self.pointer:]
            self.mergingroad[key] = pd.read_csv(merging_road_path)[key].to_numpy()[self.pointer:]

        return [self.mainroad, self.mergingroad]


