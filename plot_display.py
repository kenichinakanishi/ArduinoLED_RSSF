from ring_init import *
from random_walk import *
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def plot_display(led_info):
    for led in led_info:
        if led['number'] in led_travel:
            color = [0.9, 0.2, 0.3]
            led['color'] = color
            led['255color'] = color * 255
        plt.plot(led['x'], led['y'], 'o', color=led['color'])
    pylab.plot(x, y)
    plt.show()


def led_reset(led_info):
    for led in led_info:
        led['color'] = [0, 0, 0]
        led['255color'] = [0, 0, 0]

    return led_info


def set_led_list(led_info, list, color=[0, 0, 0]):
    """
    sets the leds in list to be a specific colour
    :param color:
    :param led_info:
    :param list:
    :return:
    """
    for i in list:
        led = led_info[i]
        # led['color'] = color/255
        led['255color'] = color
    return led_info


def send_imgdata(string, ard):
    """
    This simply sends the image data to the arduino (ard)
    :param string: produced using gen_imgdata()
    :param ard: set up using ard = serial.Serial('COM3', 500000, timeout=4)
    :return:
    """
    ard.write(string)
