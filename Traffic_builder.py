import numpy as np
from vehicle import Car


def manual(coordinates):
    data = [0] * 6
    cars = []
    #  with params = 1 for all
    # data[0] = {"id": 0, "t0": 2, "v": 10, "road": 1}
    # data[1] = {"id": 1, "t0": 3.2, "v": 5, "road": 0}
    # data[2] = {"id": 2, "t0": 4.5, "v": 10, "road": 1}
    # data[3] = {"id": 3, "t0": 5.5, "v": 15, "road": 0}

    data[0] = {"id": 0, "t0": 2, "v": 10, "road": 1}
    data[1] = {"id": 1, "t0": 3.2, "v": 5, "road": 0}
    data[2] = {"id": 2, "t0": 4.5, "v": 10, "road": 1}
    data[3] = {"id": 3, "t0": 5.5, "v": 12, "road": 0}
    data[4] = {"id": 4, "t0": 6.5, "v": 14, "road": 1}
    data[5] = {"id": 5, "t0": 7.8, "v": 15, "road": 1}

    for i in range(0, len(data)):
        entity = Car(coordinates, data[i]["id"], data[i]["t0"], data[i]["road"], 5,
                     'bicyclemodel')
        if data[i]['road'] == 0:
            entity.set_states(coordinates.mainroad['X_C'][0],
                              coordinates.mainroad['Y_C'][0],
                              coordinates.mainroad['Psi_Int'][0], data[i]['v'])
        else:
            entity.set_states(coordinates.mergingroad['X_C'][0],
                              coordinates.mergingroad['Y_C'][0],
                              coordinates.mergingroad['Psi_Int'][0], data[i]['v'])
        cars = np.append(cars, entity)

    return cars

