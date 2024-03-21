import numpy as np
from vehicle import Car


def manual(coordinates, N, total_num_cars):
    data = [0] * total_num_cars
    cars = []

    #  with params = 1 for all
    data[0] = {"id": 0, "t0": 4.5, "v": 5, "road": 1}
    data[1] = {"id": 1, "t0": 5.4, "v": 15, "road": 0}
    data[2] = {"id": 2, "t0": 6.8, "v": 14.5, "road": 1}
    data[3] = {"id": 3, "t0": 8, "v": 15, "road": 1}
    data[4] = {"id": 4, "t0": 4.5 + 6.5, "v": 5, "road": 0}
    data[5] = {"id": 5, "t0": 5.4 + 6.5, "v": 15, "road": 1}
    data[6] = {"id": 6, "t0": 6.8 + 6.5, "v": 14.5, "road": 0}
    data[7] = {"id": 7, "t0": 8 + 6.5, "v": 15, "road": 1}
    data[8] = {"id": 8, "t0": 27, "v": 9, "road": 1}
    data[9] = {"id": 9, "t0": 28.2, "v": 9, "road": 0}
    data[10] = {"id": 10, "t0": 29.5, "v": 9, "road": 1}
    data[11] = {"id": 11, "t0": 30.5, "v": 9, "road": 0}
    data[12] = {"id": 12, "t0": 31.5, "v": 9, "road": 0}
    data[13] = {"id": 13, "t0": 32.8, "v": 9, "road": 1}
    data[14] = {"id": 14, "t0": 33.9, "v": 9, "road": 0}
    data[15] = {"id": 15, "t0": 35, "v": 9, "road": 1}
    data[16] = {"id": 16, "t0": 36, "v": 9, "road": 0}
    data[17] = {"id": 17, "t0": 37.2, "v": 9, "road": 1}
    data[18] = {"id": 18, "t0": 38.4, "v": 9, "road": 1}
    data[19] = {"id": 19, "t0": 40, "v": 9, "road": 0}
    data[20] = {"id": 20, "t0": 42, "v": 8.5, "road": 1}
    data[21] = {"id": 21, "t0": 43, "v": 9, "road": 0}
    data[22] = {"id": 22, "t0": 44, "v": 10, "road": 1}
    data[23] = {"id": 23, "t0": 45.2, "v": 9, "road": 0}
    data[24] = {"id": 24, "t0": 45.8, "v": 10, "road": 1}
    data[25] = {"id": 25, "t0": 46.6, "v": 9, "road": 0}
    data[26] = {"id": 26, "t0": 47.4, "v": 11, "road": 1}
    data[27] = {"id": 27, "t0": 48.5, "v": 9, "road": 1}
    data[28] = {"id": 28, "t0": 49.2, "v": 11, "road": 0}

    for i in range(0, len(data)):
        if data[i]["road"] == 0:
            x0 = coordinates.mainroad['X_C'][0] + np.random.uniform(-0.5, 0.5)
            y0 = coordinates.mainroad['Y_C'][0] + np.random.uniform(-0.5, 0.5)
            psi0 = coordinates.mainroad['Psi_Int'][0] + 0.1 * np.random.uniform(-1,1)
            v0 = data[i]['v'] + np.random.uniform(-1, 1)
        else:
            x0 = coordinates.mergingroad['X_C'][0] + np.random.uniform(-0.5, 0.5)
            y0 = coordinates.mergingroad['Y_C'][0] + np.random.uniform(-0.5, 0.5)
            psi0 = coordinates.mergingroad['Psi_Int'][0] + 0.1 * np.random.uniform(-1,1)
            v0 = data[i]['v'] + np.random.uniform(-1, 1)

        entity = Car(coordinates, data[i]["id"], data[i]["t0"] + np.random.uniform(-0.25,0.25),
                      data[i]["road"], N, 'bicyclemodel',x0, y0, psi0, v0)
        cars = np.append(cars, entity)
    return cars

