import numpy as np
import casadi as cd
from circle_fit import taubinSVD
from scipy.integrate import solve_ivp


class Control:
    def __init__(self, dt: float, type: str, N: int):
        self.type = type
        self. N = N
        self.dt = dt
        if self.type == type:
            self.dt = dt


    def motion(self, dynamics, state, Input, timespan, teval):
        y0 = np.append(state, Input)
        sol = solve_ivp(dynamics, timespan, y0, method='DOP853', t_eval=[teval], atol=1e-6)
        x = np.reshape(sol.y[0:len(state)], len(state))
        return x


    def dynamics(self,t,x):

        dx = [0] * 6
        dx[0] = x[3] * np.cos(x[2])
        dx[1] = x[3] * np.sin(x[2])
        dx[2] = (x[3] * x[5])/(2 * 0.39)
        dx[3] = x[4]
        dx[4] = 0
        dx[5] = 0

        return dx


    def mpc_exec(self, ego, x_init, params):

        ego.referencegenerator()

        Ref_Rightlane = ego.Ref_Rightlane
        Ref_Leftlane = ego.Ref_Leftlane
        Ref_Centerlane = ego.Ref_Centerlane
        Psi_ref = ego.Ref_Psi

        x0 = x_init[0]
        y0 = x_init[1]
        psi0 = x_init[2]
        v0 = x_init[3]

        opti = cd.Opti()
        X = opti.variable(ego.statesnumber, ego.N + 1)
        u = opti.variable(ego.Inputnumber + 2, ego.N)
        s = opti.variable(4, ego.N)

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
            (psi_ref - ego.last_psi_ref_path + np.pi) % (2 * np.pi) - np.pi + ego.last_psi_ref_path
            for psi_ref in Psi_ref_path]
        ego.last_psi_ref_path = Psi_ref[0]
        for k in range(0, ego.N):
            if (Rr >= 1200 or Rl >= 1200) or abs(np.sin(Psi_ref[0])) <= 0.05:
                if y0 - Ref_Rightlane[0][1] >= 0:
                    b1 = X[1, k] - Ref_Rightlane[0][1]
                    lfb1 = X[3, k] * cd.sin(X[2, k]) + params[0] * b1
                    l2fb1 = params[0] * (lfb1 - params[0] * b1) + params[1] * lfb1
                    lgu1 = cd.sin(X[2, k])
                    lgdelta1 = X[3, k] * cd.cos(X[2, k]) * (X[3, k]) / (ego.lf + ego.lr)
                    opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[0, k] >= 0)

                    b2 = -X[1, k] + Ref_Leftlane[0][1]
                    lfb2 = -X[3, k] * cd.sin(X[2, k]) + params[2] * b2
                    l2fb2 = params[2] * (lfb2 - params[2] * b2) + params[3] * lfb2
                    lgu2 = -cd.sin(X[2, k])
                    lgdelta2 = -X[3, k] * cd.cos(X[2, k]) * (X[3, k]) / (ego.lf + ego.lr)
                    opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] + s[1, k] >= 0)

                else:
                    b1 = X[1, k] - Ref_Leftlane[0][1]
                    lfb1 = X[3, k] * cd.sin(X[2, k]) + params[4] * b1
                    l2fb1 = params[4] * (lfb1 - params[4] * b1) + params[5] * lfb1
                    lgu1 = cd.sin(X[2, k])
                    lgdelta1 = X[3, k] * cd.cos(X[2, k]) * (X[3, k]) / (ego.lf + ego.lr)
                    opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[0, k] >= 0)

                    b2 = -X[1, k] + Ref_Rightlane[0][1]
                    lfb2 = -X[3, k] * cd.sin(X[2, k]) + params[6] * b2
                    l2fb2 = params[6] * (lfb2 - params[6] * b2) + params[7] * lfb2
                    lgu2 = -cd.sin(X[2, k])
                    lgdelta2 = -X[3, k] * cd.cos(X[2, k]) * (X[3, k]) / (ego.lf + ego.lr)
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
                    X[2, k]) + xcr * cd.sin(X[2, k]))) / ((ego.lf + ego.lr) * cd.sqrt(
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
                        X[0, k] * cd.cos(X[2, k]) - xcl * cd.cos(X[2, k]) + X[
                    1, k] * cd.sin(X[2, k]) - ycl * cd.sin(X[2, k]))) / cd.sqrt(
                    (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2)) \
                        - X[3, k] * cd.cos(X[2, k]) * ((params[10] * (X[0, k] - xcl)) / cd.sqrt(
                    (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2) + (X[3, k] * cd.cos(
                    X[2, k])) / cd.sqrt((X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2) \
                                                       - (X[3, k] * (2 * X[0, k] - 2 * xcl) * (
                                X[0, k] * cd.cos(X[2, k]) - xcl * cd.cos(X[2, k]) + X[
                            1, k] * cd.sin(X[2, k]) - ycl * cd.sin(X[2, k]))) / (2 * (
                                (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2) ** (3 / 2))) \
                        - X[3, k] * cd.sin(X[2, k]) * ((params[10] * (X[1, k] - ycl)) / cd.sqrt(
                    (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2) + (X[3, k] * cd.sin(
                    X[2, k])) / cd.sqrt((X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2) \
                                                       - (X[3, k] * (2 * X[1, k] - 2 * ycl) * (
                                X[0, k] * cd.cos(X[2, k]) - xcl * cd.cos(X[2, k]) + X[
                            1, k] * cd.sin(X[2, k]) - ycl * cd.sin(X[2, k]))) / (2 * (
                                (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2) ** (3 / 2)))

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
                                   (ego.lf + ego.lr) * cd.sqrt(
                               (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2))

                lgu2 = -(X[0, k] * cd.cos(X[2, k]) - xcl * cd.cos(X[2, k]) + X[1, k] * cd.sin(
                    X[2, k]) - ycl * cd.sin(X[2, k])) / cd.sqrt(
                    (X[0, k] - xcl) ** 2 + (X[1, k] - ycl) ** 2)

                # opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] >= 0)
                # opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] >= 0)
                opti.subject_to(l2fb2 + lgu2 * u[0, k] + lgdelta2 * u[1, k] + s[0, k] >= 0)
                opti.subject_to(l2fb1 + lgu1 * u[0, k] + lgdelta1 * u[1, k] + s[1, k] >= 0)

            if k == 0:
                b11 = b1
                lfb11 = lfb1
                l2fb11 = l2fb1
                lgu11 = lgu1
                lgdelta11 = lgdelta1

                b21 = b2
                lfb21 = lfb2
                l2fb21 = l2fb2
                lgu21 = lgu2
                lgdelta21 = lgdelta2

        if ego.road == 0:
            curr_states = x_init[6:10]
            for k in range(0, ego.N):
                a = 1.5
                b = 1
                xip, yip, psi_ip, vip = curr_states

                b3 = (X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2
                lfb3 = (params[12] * ((X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2) +
                        (X[3, k] * cd.cos(X[2, k]) * (2 * X[0, k] - 2 * xip)) / a -
                        (vip * cd.cos(psi_ip) * (2 * X[0, k] - 2 * xip)) / a +
                        (X[3, k] * cd.sin(X[2, k]) * (2 * X[1, k] - 2 * yip)) / b -
                        (vip * cd.sin(psi_ip) * (2 * X[1, k] - 2 * yip)) / b)
                lgb3u = -2 * X[3, k]
                lgb3delta = 0

                opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
                curr_states = self.motion(self.dynamics, curr_states, [0, 0], [0, self.dt], self.dt)
                if k == 0:
                    b31 = b3
                    lfb31 = lfb3
                    lgb3u1 = lgb3u
                    lgb3delta1 = lgb3delta


            curr_states = x_init[10:]
            for k in range(0, ego.N):
                xic, yic, psi_ic, vic = curr_states
                R_i = cd.sqrt(x0**2 + (y0-1000)**2)
                R_c = cd.sqrt(xic**2 + (yic-100)**2)
                b01 = 1000
                b00 = 100
                delta = 0
                theta_0_i = -1.4672603502370494
                theta_0_ic = -0.51
                const = 1.52 -1.4672603502370494
                b4 =  (R_c * (theta_0_ic - np.arctan((yic - b00)/xic)) - R_i * (theta_0_i - np.arctan((y0 - b01)/ x0))
                       - (theta_0_i -np.arctan((y0 - b01)/ x0) )/ const * v0 - delta)

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
                lgb4u = -(theta_0_i + cd.atan2(b01 - X[1, k] ,X[0, k])) / const
                lgb4delta = 0
                # print(b4)

                opti.subject_to(lfb4 + lgb4u * u[0, k] + lgb4delta * u[1, k] + s[3, k] >= 0)
                curr_states = self.motion(self.dynamics, curr_states, [0, 0], [0, self.dt], self.dt)
                if k == 0:
                    b41 = b4
                    lfb41 = lfb4
                    lgb4u1 = lgb4u
                    lgb4delta1 = lgb4delta

        else:
            curr_states = x_init[6:10]
            for k in range(0, ego.N):
                a = 1.5
                b = 1
                xip, yip, psi_ip, vip = curr_states
                b3 = (X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2
                lfb3 = (params[14] * ((X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2) +
                        (X[3, k] * cd.cos(X[2, k]) * (2 * X[0, k] - 2 * xip)) / a -
                        (vip * cd.cos(psi_ip) * (2 * X[0, k] - 2 * xip)) / a +
                        (X[3, k] * cd.sin(X[2, k]) * (2 * X[1, k] - 2 * yip)) / b -
                        (vip * cd.sin(psi_ip) * (2 * X[1, k] - 2 * yip)) / b)
                lgb3u = -2 * X[3, k]
                lgb3delta = 0

                opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
                curr_states = self.motion(self.dynamics, curr_states, [0, 0], [0, self.dt], self.dt)
                if k == 0:
                    b31 = b3
                    lfb31 = lfb3
                    lgb3u1 = lgb3u
                lgb3delta1 = lgb3delta

            curr_states = x_init[10:]
            for k in range(0, ego.N):
                xic, yic, psi_ic, vic = curr_states
                R_i = cd.sqrt(x0**2 + (y0-100)**2)
                R_c = cd.sqrt(x0**2 + (y0-1000)**2)
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

                term4 = params[15] * (delta + R_i * (theta_0_i + cd.atan2(b01 - X[1, k], X[0, k])) - R_c * (
                        theta_0_ic + cd.atan2(b00 - yic, xic)) + (
                                              X[3, k] * (theta_0_i + cd.atan2(b01 - X[1, k], X[0, k]))) / const)

                term5_numerator = X[3, k] * cd.cos(X[2, k]) * (b01 - X[1, k]) * (X[3, k] + R_i * const)
                term5_denominator = const * (b01 ** 2 - 2 * b01 * X[1, k] + X[0, k] ** 2 + X[1, k] ** 2)
                term5 = term5_numerator / term5_denominator
                lfb4 = term1 + term2 + term3 - term4 + term5
                lgb4u = -(theta_0_i + cd.atan2(b01 - X[1, k], X[0, k])) / const
                lgb4delta = 0
                # print(b4)

                opti.subject_to(lfb4 + lgb4u * u[0, k] + lgb4delta * u[1, k] + s[3, k] >= 0)
                curr_states = self.motion(self.dynamics, curr_states, [0, 0], [0, self.dt], self.dt)

                if k == 0:
                    b41 = b4
                    lfb41 = lfb4
                    lgb4u1 = lgb4u
                    lgb4delta1 = lgb4delta

        for k in range(0, ego.N):
            b = X[3, k] - ego.vmin
            lfb = 1 * b
            lfb = 1 * b
            lgu = 1
            opti.subject_to(lfb + lgu * u[0, k] >= 0)

            b = -X[3, k] + ego.vmax
            lfb = 1 * b
            lfb = 1 * b
            lgu = -1
            opti.subject_to(lfb + lgu * u[0, k] >= 0)

        # Define the cost function
        cost = 0
        diag_elements_u = [params[16], params[17], params[18],  params[19]]
        # diag_elements_u = [0.0001, 0.0001, 1, 0.0001]
        # diag_elements_u = [1, 1, 100, 100]
        u_ref = np.zeros((ego.Inputnumber + 2, ego.N))
        # u_ref[0] = [5,5]
        normalization_factor = [max(-ego.umin, ego.umax), max(ego.steermax, -ego.steermin),
                                200, 0.4]
        for i in range(ego.Inputnumber + 2):
            for h in range(ego.N):
                cost += 0.5 * diag_elements_u[i] * ((u[i, h] - u_ref[i][h]) / normalization_factor[i]) ** 2
                cost += 10 ** 8 * (s[0, h]) ** 2
                cost += 10 ** 8 * (s[1, h]) ** 2
                cost += 10 ** 8 * (s[2, h]) ** 2
                cost += 10 ** 8 * (s[3, h]) ** 2

        opti.subject_to(ego.umin <= u[0, :])
        opti.subject_to(u[0, :] <= ego.umax)
        eps3 = 1
        for k in range(0, ego.N):
            V = (X[3, k] - 15) ** 2
            lfV = eps3 * V
            lgu = 2 * (X[3, k] - 15)
            opti.subject_to(lfV + lgu * u[0, k] - u[2, k] <= 0)

            V = (X[2, k] - Psi_ref_path_continous[0]) ** 2
            lfV = 10*eps3 * V
            lgdelta = 2 * (X[2, k] - Psi_ref_path_continous[0]) * X[3, k] / (ego.lr + ego.lf)
            opti.subject_to(lfV + lgdelta * u[1, k] - u[3, k] <= 0)
        # print(psi0, Psi_ref_path[0], Psi_ref_path_continous[0])
        # print(psi0 - Psi_ref_path[0])
        opti.subject_to(ego.steermin <= u[1, :])
        opti.subject_to(u[1, :] <= ego.steermax)

        opti.subject_to(X[:, 0] == x_init[0:4])  # initialize states
        timespan = [0, self.dt]

        for h in range(ego.N):  # initial guess
            opti.set_initial(X[:, h], [Ref_Centerlane[h][0], Ref_Centerlane[h][1], Psi_ref[h], v0])

        for k in range(ego.N):
            state = []
            Input = []
            for j in range(ego.statesnumber):
                state.append(X[j, k])
            for j in range(0, ego.Inputnumber):
                Input.append(u[j, k])
            state = ego.rk4(timespan, state, Input, 1)
            for j in range(ego.statesnumber):
                opti.subject_to(X[j, k + 1] == state[j])

        try:
            opts = {}
            opts['print_time'] = False
            opts['ipopt.print_level'] = False
            opti.solver('ipopt', opts)
            opti.minimize(cost)
            sol = opti.solve()

            if ego.N > 1:
                # if self.agent.road == 0:
                #     print(sol.value(b41),params[12])
                # else:
                #     print(sol.value(b41), params[13])
                # RightCBF = sol.value(l2fb11 + lgu11 * sol.value(u)[0, 0] + lgdelta11 * sol.value(u)[1, 0])
                # LeftCBF = sol.value(l2fb21 + lgu21 * sol.value(u)[0, 0] + lgdelta21 * sol.value(u)[1, 0])
                # # RearendCBF = sol.value(lfb31 + lgb3u1 * sol.value(u)[0, 0] + lgb3delta1 * sol.value(u)[1, 0])
                # MergingCBF = sol.value(lfb41 + lgb4u1 * sol.value(u)[0, 0] + lgb4delta1 * sol.value(u)[1, 0])
                #
                # # right_cbf = b1
                # # # left_cbf = b2
                # self.agent.RightCBF.append(RightCBF)
                # self.agent.LeftCBF.append(LeftCBF)
                # # self.agent.RearendCBF.append(RearendCBF)
                # self.agent.MergingCBF.append(MergingCBF)
                # self.agent.LeftOrg.append(sol.value(b11))
                # self.agent.RightOrg.append(sol.value(b21))
                # self.agent.MergingOrg.append(sol.value(b41))
                # # self.agent.RearendOrg.append(sol.value(b31))
                # self.agent.accdata.append(sol.value(u)[0, 0])
                # self.agent.steerdata.append(sol.value(u)[1, 0])
                # self.agent.obj.append(sol.value(cost))
                ego.s_vars = sol.value(s)[:, 0]
                if np.any(ego.s_vars > 0.1):  # or LeftCBF <= 0 or RightCBF <= 0:
                    ego.s_vars = [min(sol.value(s)[i, 0], 0.1) for i in range(len(sol.value(s)[:, 0]))]
                    # self.agent.s_vars = sol.value(s)[2, 0]
                    return "No solution found", np.array([-5.66, 0])
                else:
                    return "solution found", sol.value(u)[:2, 0]
            else:

                # RightCBF = sol.value(l2fb11 + lgu11 * sol.value(u[0]) + lgdelta11 * sol.value(u[1]))
                # LeftCBF = sol.value(l2fb21 + lgu21 * sol.value(u[0]) + lgdelta21 * sol.value(u[1]))
                # RearendCBF = sol.value(lfb31 + lgb3u1 * sol.value(u[0]) + lgb3delta1 * sol.value(u[1]))
                # MergingCBF = sol.value(lfb41 + lgb4u1 * sol.value(u[0]) + lgb4delta1 * sol.value(u[1]))
                #
                # self.agent.RightCBF.append(RightCBF)
                # self.agent.LeftCBF.append(LeftCBF)
                # self.agent.RearendCBF.append(RearendCBF)
                # self.agent.MergingCBF.append(MergingCBF)
                # self.agent.LeftOrg.append(sol.value(b11))
                # self.agent.RightOrg.append(sol.value(b21))
                # self.agent.MergingOrg.append(sol.value(b31))
                # self.agent.RearendOrg.append(sol.value(b41))
                # self.vehicle.RightCBF.append(sol.value(b11))
                # self.vehicle.LeftCBF.append(sol.value(b21))
                ego.s_vars = sol.value(s)
                if np.any(sol.value(s) > 0.1):  # or LeftCBF <= 0 or RightCBF <= 0:
                    ego.s_vars = [0.1] * 4
                    return "No solution found", np.array([-5.66, 0])
                else:
                    return "solution found", sol.value(u[:2])

        except:
            # self.vehicle.RightCBF.append(sol.value(l2fb1 + lgu1 * -5.66 + lgdelta1 * 0))
            # self.vehicle.LeftCBF.append(sol.value(l2fb2 + lgu2 * -5.66 + lgdelta2 * 0))
            # b11 = -5
            # b21 = -5
            # self.agent.RightCBF.append(b11)
            # self.agent.LeftCBF.append(b21)
            return "No solution found", np.array([-5.66, 0])

        # if ego.N > 1:
        #     if ego.id == 0:
        #         # reward += -0.05 * (sol.value(u)[0, 0] - 0) ** 2 / 5 ** 2
        #         # reward += -0.05 * (sol.value(u)[1, 0] - 0) ** 2 / 0.45 ** 2
        #         # reward += -0.3 * (v0 - 15) ** 2 / 15 ** 2
        #         # reward += -0.3 * (psi0 - Psi_ref_path_continous[0]) ** 2 / 3.14 ** 2
        #         Dist2Centerarray = np.array([])
        #         for k in range(0, len(Ref_Centerlane_x)):
        #             Dist2Centerarray = np.append(Dist2Centerarray,
        #                                          (x0 - Ref_Centerlane_x[k]) ** 2 + (y0 - Ref_Centerlane_y[k]) ** 2)
        #         Distance2CentraLane = np.min(Dist2Centerarray)
        #         reward += -0.5 * Distance2CentraLane / 6.25
        #         reward += -0.1 * (v0 - 15) ** 2 / 15 ** 2
        #         reward += -0.3 * (psi0 - Psi_ref_path_continous[0]) ** 2 / 3.14 ** 2
        #
        #     # reward += -0.15 * self.agent.s_vars[0] / 0.01
        #     # reward += -0.15 * self.agent.s_vars[1] / 0.01
        #     # reward += -0.15 * self.agent.s_vars[2] / 0.01
        #     # reward += -0.15 * self.agent.s_vars[3] / 0.01
        #
        #     # print(sol.value(b41),sol.value(lfb41 + lgb4u1 * sol.value(u)[0, 0] + lgb4delta1 * sol.value(u)[1, 0]) )
        #
        #     # print(np.arctan((y0-100)/x0))
        #     # print(np.sqrt(x0**2 + (y0-1000)**2))
        #
        #     # print(np.sqrt(x0**2 + (y0-1000)**2)*(-1.4672603502370494 - np.arctan((y0-1000)/x0)))
        #     # print(np.sqrt(x0**2 + (y0-100)**2)*(-0.51 - np.arctan((y0-100)/x0)))
        #
        #     # RightCBF = sol.value(l2fb11 + lgu11 * sol.value(u)[0, 0] + lgdelta11 * sol.value(u)[1, 0])
        #     # LeftCBF = sol.value(l2fb21 + lgu21 * sol.value(u)[0, 0] + lgdelta21 * sol.value(u)[1, 0])
        #     # RearendCBF = sol.value(lfb31 + lgb3u1 * sol.value(u)[0, 0] + lgb3delta1 * sol.value(u)[1, 0])
        #     # ego.obj.append(sol.value(cost))
        #     # MergingCBF = sol.value(lfb41 + lgb4u1 * sol.value(u)[0, 0] + lgb4delta1 * sol.value(u)[1, 0])
        #     # right_cbf = b1
        #     # left_cbf = b2
        #     # ego.RightCBF.append(RightCBF)
        #     # ego.LeftCBF.append(LeftCBF)
        #     # ego.RearendCBF.append(RearendCBF)
        #     # ego.MergingCBF.append(MergingCBF)
        #     # ego.LeftOrg.append(sol.value(b11))
        #     # ego.RightOrg.append(sol.value(b21))
        #     # ego.RearendOrg.append(sol.value(b31))
        #     # ego.MergingOrg.append(sol.value(b41))
        #     # if sol.value(b41) < 0:
        #     #     stop = 1
        #     # ego.accdata.append(sol.value(u)[0, 0])
        #     # ego.steerdata.append(sol.value(u)[1, 0])
        #     # ego.s_vars = sol.value(s)[2, 0]
        #
        #     if np.any(sol.value(s) > 0.1):  # or LeftCBF <= 0 or RightCBF <= 0:
        #         ego.s_vars = 0.1
        #         # ego.s_vars = sol.value(s)[2, 0]
        #         return "No solution found", np.array([-5.66, 0]), reward
        #     else:
        #         return "solution found", sol.value(u)[:2, 0], reward
        # else:
        #     #
        #     # RightCBF = sol.value(l2fb11 + lgu11 * sol.value(u[0]) + lgdelta11 * sol.value(u[1]))
        #     # LeftCBF = sol.value(l2fb21 + lgu21 * sol.value(u[0]) + lgdelta21 * sol.value(u[1]))
        #     # RearendCBF = sol.value(lfb31 + lgb3u1 * sol.value(u[0]) + lgb3delta1 * sol.value(u[1]))
        #     # MergingCBF = sol.value(lfb41 + lgb4u1 * sol.value(u[0]) + lgb4delta1 * sol.value(u[1]))
        #
        #     # ego.RightCBF.append(RightCBF)
        #     # ego.LeftCBF.append(LeftCBF)
        #     # ego.RearendCBF.append(RearendCBF)
        #     # # ego.MergingCBF.append(MergingCBF)
        #     # ego.LeftOrg.append(sol.value(b11))
        #     # ego.RightOrg.append(sol.value(b21))
        #     # ego.RearendOrg.append(sol.value(b31))
        #     # # ego.RearendOrg.append(sol.value(b41))
        #     # self.vehicle.RightCBF.append(sol.value(b11))
        #     # self.vehicle.LeftCBF.append(sol.value(b21))
        #     # ego.s_vars = sol.value(s[2])
        #     if np.any(sol.value(s) > 0.1):  # or LeftCBF <= 0 or RightCBF <= 0:
        #         return "No solution found", np.array([-5.66, 0]), reward
        #     else:
        #         return "solution found", sol.value(u[:2]), reward

