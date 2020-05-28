from ring_init import *

# Defining the LED indexes around each sensor for later use
sensor1 = [133, 134, 135, 180, 181, 182, 234, 235, 236]
sensor2 = [145, 146, 147, 194, 195, 249, 250, 251]
sensor3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
sensor4 = [156, 157, 158, 207, 208, 209, 264, 265, 266]
sensor5 = [168, 169, 170, 220, 221, 222, 279, 280, 281]
spot_middle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
spot_1 = [133, 134, 135, 180, 181, 182, 234, 235, 236]
spot_2 = [145, 146, 147, 194, 195, 249, 250, 251]
spot_3 = [156, 157, 158, 207, 208, 209, 264, 265, 266]
spot_4 = [168, 169, 170, 220, 221, 222, 279, 280, 281]


# Random Walk Code
def rand_walk(num, qdisplay):
    import random
    import numpy as np
    import math
    x_step = np.zeros(num)
    y_step = np.zeros(num)
    step = 4
    y_jitter = 1.5
    y_step[0] = random.randint(100, 400)
    x_step[0] = 1
    for i in range(1, 10):
        val = random.randint(1, 8)
        if val == 1:
            x_step[i] = x_step[i - 1]
            y_step[i] = y_step[i - 1] - step * y_jitter
        elif val == 3:
            x_step[i] = x_step[i - 1]
            y_step[i] = y_step[i - 1] + step * y_jitter
        else:
            x_step[i] = x_step[i - 1] + step
            y_step[i] = y_step[i - 1]
    for i in range(10, num):
        val = random.randint(1, 5)
        if val == 1:
            x_step[i] = x_step[i - 1]
            y_step[i] = y_step[i - 1] - step * y_jitter
            if y_step[i] not in range(0, 500):
                break
        elif val == 2:
            x_step[i] = x_step[i - 1] - step
            y_step[i] = y_step[i - 1]
            if x_step[i] not in range(0, 500):
                break
        elif val == 3:
            x_step[i] = x_step[i - 1]
            y_step[i] = y_step[i - 1] + step * y_jitter
            if y_step[i] not in range(0, 500):
                break
        else:
            x_step[i] = x_step[i - 1] + step
            y_step[i] = y_step[i - 1]
            # check if defect site hit
            point = point2led([x_step[i], y_step[i]])
            sensor_1 = qdisplay[4]
            sensor_2 = qdisplay[3]
            sensor_middle = qdisplay[2]
            sensor_3 = qdisplay[0]
            sensor_4 = qdisplay[1]
            if sensor_middle is True and point in spot_middle:
                step = step * -1
                print('hit Middle')
            elif sensor_1 is True and point in spot_1:
                step = step * -1
                print('hit point 1')
            elif sensor_2 is True and point in spot_2:
                step = step * -1
                print('hit point 2')
            elif sensor_3 is True and point in spot_3:
                step = step * -1
                print('hit point 3')
            elif sensor_4 is True and point in spot_4:
                step = step * -1
                print('hit point 4')
            elif x_step[i] not in range(0, 500):
                break
    arraylength = x_step.size
    index = []
    for i in range(1, arraylength):
        if x_step[i] == 0:
            index.append(i)
        elif math.sqrt(abs(250 - 100) ** 2 + abs(250 - 100 ** 2)) > 250:
            index.append(i)
    x_step = np.delete(x_step, index)
    y_step = np.delete(y_step, index)
    # create list of LEDS to travel through
    led_travel = []
    convert_length = x_step.size
    for m in range(0, convert_length):
        point = point2led([x_step[m], y_step[m]])
        led_travel.append(point)
    # delete repetition and unnecessary jitter
    delete_index = []
    for m in range(1, convert_length):
        if led_travel[m] == led_travel[m - 1]:
            delete_index.append(m)
        elif abs(led_travel[m - 3] - led_travel[m - 2]) + abs(led_travel[m - 2] - led_travel[m - 1]) + abs(
                led_travel[m - 1] - led_travel[m]) <= 3:
            delete_index.append(m)
    led_travel = np.delete(led_travel, delete_index)
    x_step = np.delete(x_step, delete_index)
    y_step = np.delete(y_step, delete_index)
    return [x_step, y_step, led_travel]
