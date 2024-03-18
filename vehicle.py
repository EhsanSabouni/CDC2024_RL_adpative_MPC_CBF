import numpy as np
from scipy.integrate import odeint,solve_ivp,ode
import matplotlib.pyplot as plt
import casadi as cd
from circle_fit import taubinSVD
import time
class Car:
    def __init__(self, coordinates, id, t0, road: int, N:int, method: str):
        self.id = id
        self.t0 = t0
        self.road = road
        self.lf = 0.39
        self.lr = 0.39
        self.x = 0
        self.y = 0
        self.v = 0
        self.psi = 0
        self.acc = 0
        self.steer = 0
        self.method = 'bicyclemodel'
        self.vmin = 0
        self.vmax = 35
        self.umax = 4
        self.umin = -5
        self.steermax = cd.pi/4
        self.steermin = -cd.pi/4
        self.Ref_Rightlane = []
        self.Ref_Leftlane = []
        self.Ref_Centerlane = []
        self.coordinates = coordinates
        self.Psi_ref = []
        self.RightCBF = []
        self.LeftCBF = []
        self.MergingCBF = []
        self.RearendCBF = []
        self.RightOrg = []
        self.LeftOrg = []
        self.MergingOrg = []
        self.RearendOrg = []
        self.obj = []
        self.accdata = []
        self.steerdata = []
        self.NumPoints = 30
        self.s_vars = []
        self.desired_speed = 0
        self.last_psi_ref_path = 0
        self.N = N
        self.energy = 0
        self.time = 0
        self.fuel = 0
        self.ip_states = [-1000, 0, 0, 0]
        self.ic_states = [-1000, 0, 0, 0]
        if method == 'bicyclemodel':
           self.Inputnumber = 2
           self.statesnumber = 4

    def dynamics(self,t,x):
        if self.method == 'bicyclemodel':
            dx = [0] * 6
            dx[0] = x[3] * np.cos(x[2])
            dx[1] = x[3] * np.sin(x[2])
            dx[2] = (x[3] * x[5])/(self.lf + self.lr)
            dx[3] = x[4]
            dx[4] = 0
            dx[5] = 0
        return dx
    
    def set_states(self, x, y, psi, v):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v

    def get_states(self):
        return [self.x, self.y, self.psi, self.v]


    def motion(self, dynamics, state, Input, timespan, teval):
        y0 = np.append(state, Input)
        sol = solve_ivp(dynamics, timespan, y0, method = 'DOP853', t_eval=[teval], atol=1e-6)
        x = np.reshape(sol.y[0:len(state)], len(state))
        return x


    def rk4(self, t, state, Input, n ):
        state = np.append(state, Input)
        # Calculating step size
        # x0 = np.append(state, Input)
        h = np.array([(t[-1] - t[0]) / n])
        t0 = t[0]
        for i in range(n):
            k1 = np.array(self.dynamics(t0, state))
            k2 = np.array(self.dynamics((t0 + h / 2), (state + h * k1 / 2)))
            k3 = np.array(self.dynamics((t0 + h / 2), (state + h * k2 / 2)))
            k4 = np.array(self.dynamics((t0 + h), (state + h * k3)))
            k = np.array(h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
            # k = np.array(h * (k1))
            xn = state + k
            state = xn
            t0 = t0 + h

        return xn[0:self.statesnumber]

    def metric_update(self, dt):
        self.energy = self.energy + 0.5 * self.acc ** 2 * dt
        if self.acc >= 0:
            b = [0.1569, 0.02450, -0.0007415, 0.00005975]
            c = [0.07224, 0.09681, 0.001075]
            self.fuel = self.fuel + dt * (self.acc * (c[0] + c[1]*self.v + c[2]*self.v**2)
                                          +(b[0] + b[1]*self.v + b[2]*self.v**2 + b[3]*self.v**3))
        else:
            self.fuel = self.fuel
        self.time = self.time + dt
        return None

    def referencegenerator(self):
        predictedstate = []
        Dist2Centerarray = []
        Dist2Right = []
        Dist2Left = []
        roaddata = [self.coordinates.mainroad, self.coordinates.mergingroad]
        X_C = roaddata[self.road]['X_C']
        Y_C = roaddata[self.road]['Y_C']
        X_R = roaddata[self.road]['X_R']
        Y_R = roaddata[self.road]['Y_R']
        X_L = roaddata[self.road]['X_L']
        Y_L = roaddata[self.road]['Y_L']
        PsiInt = roaddata[self.road]['Psi_Int']


        for k in range(0, len(X_C)):
            Dist2Centerarray = np.append(Dist2Centerarray, (self.x - X_C[k]) ** 2 + (self.y - Y_C[k]) ** 2)

        Index_Point_on_centerlane = np.argmin(Dist2Centerarray)
        for k in range(0, len(X_R)):
            Dist2Right = np.append(Dist2Right, (X_C[Index_Point_on_centerlane] - X_R[k]) ** 2 +
                                   (Y_C[Index_Point_on_centerlane] - Y_R[k]) ** 2)
        Index_Point_on_Rightlane = np.argmin(Dist2Right)
        for k in range(0, len(X_L)):
            Dist2Left = np.append(Dist2Left, (X_C[Index_Point_on_centerlane] - X_L[k]) ** 2 +
                                  (Y_C[Index_Point_on_centerlane] - Y_L[k]) ** 2)
        Index_Point_on_Leftlane = np.argmin(Dist2Left)

        Ref_Psi = PsiInt[Index_Point_on_centerlane:min(Index_Point_on_centerlane + self.N + 1, len(PsiInt))]

        Ref_Centerlane = []
        if Index_Point_on_centerlane + self.NumPoints <= len(X_C):
            for k in range(self.NumPoints):
                Ref_Centerlane.append([X_C[Index_Point_on_centerlane + k], Y_C[Index_Point_on_centerlane + k]])
        else:
            for k in range(len(X_C) - Index_Point_on_centerlane):
                Ref_Centerlane.append([X_C[Index_Point_on_centerlane + k], Y_C[Index_Point_on_centerlane + k]])

        Ref_Rightlane = []
        if Index_Point_on_Rightlane + self.NumPoints <= len(X_R):
            for k in range(self.NumPoints):
                Ref_Rightlane.append([X_R[Index_Point_on_Rightlane + k], Y_R[Index_Point_on_Rightlane + k]])
        else:
            for k in range(len(X_R) - Index_Point_on_Rightlane):
                Ref_Rightlane.append([X_R[Index_Point_on_Rightlane + k], Y_R[Index_Point_on_Rightlane + k]])

        Ref_Leftlane = []
        if Index_Point_on_Leftlane + self.NumPoints <= len(X_L):
            for k in range(self.NumPoints):
                Ref_Leftlane.append([X_L[Index_Point_on_Leftlane + k], Y_L[Index_Point_on_Leftlane + k]])
        else:
            for k in range(len(X_L) - Index_Point_on_Leftlane):
                Ref_Leftlane.append([X_L[Index_Point_on_Leftlane + k], Y_L[Index_Point_on_Leftlane + k]])

        self.N = min(len(Ref_Psi), self.N)
        if len(Ref_Psi) <= self.N:
            Ref_Psi = PsiInt[Index_Point_on_centerlane - 1: len(PsiInt)]

        self.Ref_Psi = Ref_Psi
        self.Ref_Rightlane = Ref_Rightlane
        self.Ref_Leftlane = Ref_Leftlane
        self.Ref_Centerlane = Ref_Centerlane

        return None



