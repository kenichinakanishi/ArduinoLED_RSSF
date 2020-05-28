# Setting up and defining LED radii:
ring_radii = [
    10.61032954,
    29.17840623,
    47.74648293,
    66.31455962,
    84.88263632,
    103.450713,
    122.0187897,
    140.5868664,
    159.1549431,
    177.7230198,
    196.2910965,
    214.8591732,
    233.4272499
]

# Defining number of LED's existing in each concentric ring
no_leds = [4,
           11,
           18,
           25,
           32,
           39,
           46,
           53,
           60,
           67,
           74,
           81,
           88]

ring_1 = list(range(0, 4))
ring_2 = list(range(4, 15))
ring_3 = list(range(15, 33))
ring_4 = list(range(33, 58))
ring_5 = list(range(58, 90))
ring_6 = list(range(90, 129))
ring_7 = list(range(129, 175))
ring_8 = list(range(175, 228))
ring_9 = list(range(228, 288))
ring_10 = list(range(288, 355))
ring_11 = list(range(355, 429))
ring_12 = list(range(429, 510))
ring_13 = list(range(510, 598))


# define distance function
def dist(p0, p1):
    import math
    return math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)


# define point to closest led function
def point2led(target):
    sorted_by_dist = sorted(led_info, key=lambda p: dist(target, p['co-ordinate']))
    closest = sorted_by_dist[0]
    closest_led = closest['number']
    return closest_led


def make_led_coordinates(radii, no_leds):
    """
    This function creates an list of dictionaries for easy assignment, with each dictionary holding information on each
    LED
    :param radii:
    :param no_leds:
    :return:
    """
    import numpy as np
    led_info = []
    n = 0
    for i in range(len(radii)):
        r = radii[i]
        for j in range(no_leds[i]):
            d = {}
            angle = 2 * np.pi * j / no_leds[i]
            x = r * np.cos(angle)
            y = -r * np.sin(angle)
            d['x'] = (x + 244)
            d['y'] = (y + 244)
            d['angle'] = angle
            d['radii'] = r
            d['number'] = n
            d['co-ordinate'] = [(x + 244), (y + 244)]
            d['color'] = [0, 0, 0]
            d['255color'] = [0, 0, 0]
            led_info.append(d)
            n += 1

    return led_info
