import matplotlib.pyplot as plt
import numpy as np
import serial
import time
from PIL import Image as im
from PIL import ImageOps as imo
from scipy import interpolate
import copy
import re
import os
import sys
import math
import random

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

spot_middle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
spot_1 = [133, 134, 135, 180, 181, 182, 234, 235, 236]
spot_2 = [145, 146, 147, 194, 195, 249, 250, 251]
spot_3 = [156, 157, 158, 207, 208, 209, 264, 265, 266]
spot_4 = [168, 169, 170, 220, 221, 222, 279, 280, 281]

# define distance function
def dist(p0, p1):
    return math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)


# define point to closest led function
def point2led(target):
    sorted_by_dist = sorted(led_info, key=lambda p: dist(target, p['co-ordinate']))
    closest = sorted_by_dist[0]
    closest_led = closest['number']
    return closest_led


# Random Walk Code
def rand_walk(num, qdisplay):
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
    m = 0
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


# Get Files from folder
def get_imlist(path):
    """  Returns a list of filenames for
    all jpg images in a directory. """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpeg')]


# Sort files into numerical order
numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def make_led_coordinates(radii, no_leds):
    '''
    This function creates an list of dictionaries for easy assignment, with each dictionary holding information on each
    LED
    :param radii:
    :param no_leds:
    :return:
    '''
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


def plot_display(led_info):
    for led in led_info:
        if led['number'] in led_travel:
            color = [0.9, 0.2, 0.3]
            led['color'] = color
            led['255color'] = color * 255
        plt.plot(led['x'], led['y'], 'o', color=led['color'])
    pylab.plot(x, y)
    plt.show()


def image2ring(led_info, imgfn, box_rad=1):
    '''
    This function take the LED list, the file name of an image and an 'averaging spec' to allocate representative
    colours of all LEDs to generate that image.
    :param led_info:
    :param imgfn:
    :param box_rad:
    :return:
    '''
    img = plt.imread(imgfn)
    img = np.asarray(img)
    img = im.fromarray(img)
    img = imo.fit(img, (488, 488), method=im.ANTIALIAS)
    img = img.rotate(-30)
    imarray = np.asarray(img)
    for led in led_info:
        ledx = led['x']
        ledy = led['y']
        color = np.average(
            np.average(
                imarray[int(ledx) - box_rad - 1:int(ledx) + box_rad, int(ledy) - box_rad - 1:int(ledy) + box_rad,:] / 255,
                0),
            0)[0:3]
        led['color'] = color
        led['255color'] = color * 255
    return led_info


def gen_imgdata(led_info, imgfn):
    '''
    This function generates the string that gets sent serially to the arduino to display an image.
    :param led_info:
    :return:
    '''
    led_info = image2ring(led_info, imgfn)
    string = ''
    for led in led_info:
        string += str(int(led['255color'][0])) + ','
        string += str(int(led['255color'][1])) + ','
        string += str(int(led['255color'][2])) + ','
    return string.encode()


def led_reset(led_info):
    for led in led_info:
        led['color'] = [0, 0, 0]
        led['255color'] = [0, 0, 0]

    return led_info


def set_led_list(led_info, list, color=[0, 0, 0]):
    '''
    sets the leds in list to be a specific colour
    :param led_info:
    :param list:
    :return:
    '''
    for i in list:
        led = led_info[i]
        # led['color'] = color/255
        led['255color'] = color
    return led_info


def send_imgdata(string, ard):
    '''
    This simply sends the image data to the arduino (ard)
    :param string: produced using gen_imgdata()
    :param ard: set up using ard = serial.Serial('COM3', 500000, timeout=4)
    :return:
    '''
    ard.write(string)


def generate_video(fns, led_info):
    '''
    takes a list of file names and compiles them into a list of images to be uploaded sequentially to the arduino using
    upload_video()
    :param fns:
    :param led_info:
    :return:
    '''
    videolist = []
    for fn in fns:
        videolist.append(gen_imgdata(led_info, fn))

    return videolist


def upload_video(imglist, arduino, loops=1, frame_delay=0.05):
    '''
    This function sends a video to the arduino frame by frame for a set number of loops
    :param imglist: the list of images created by generate video or otherwise.
    :param arduino:
    :param loops:
    :param frame_delay:
    :return:
    '''
    for i in range(loops):
        for img in imglist:
            send_imgdata(img, arduino)
            time.sleep(frame_delay)


def capture_sensvalues(ard, no_sensors=5, data_points=64, baseline_remove=True):
    '''
    Returns numpy array of sensor values taken from arduino through the serial port
    :param no_sensors: number of sensors the arduino has
    :param polls:
    :param pollrate:
    :return:
    '''
    ard.write(b'c')
    values = ard.readlines()
    # print(values)
    data = np.zeros([data_points, no_sensors])
    times = np.zeros(data_points)
    n = 0
    for i in range(data_points):
        vals = values[i * (no_sensors + 1):(i + 1) * (no_sensors + 1)]
        for j in range(no_sensors):
            data[i, j] = vals[j + 1][:-2]
        times[i] = vals[0][:-2]
    times = times * 1e-6
    if baseline_remove:
        for i in range(no_sensors):
            data[:, i] = data[:, i] / np.int(np.round(np.mean(data[:, i])))
    return times, data


def quickcapture_sensvalues(ard, no_sensors=5, data_points=64, baseline_remove=True):
    '''
    Returns numpy array of sensor values taken from arduino through the serial port
    :param no_sensors: number of sensors the arduino has
    :param polls:
    :param pollrate:
    :return:
    '''
    ard.write(b'c')
    values = ard.readlines()
    # print(values)
    data = np.zeros([data_points, no_sensors])
    n = 0
    for i in range(data_points):
        vals = values[i * (no_sensors):(i + 1) * (no_sensors)]
        for j in range(no_sensors):
            data[i, j] = vals[j][:-2]
    if baseline_remove:
        for i in range(no_sensors):
            data[:, i] = data[:, i] / np.int(np.round(np.mean(data[:, i])))
    return [0], data


def get_sensamp(ard, baseline):
    t, data = quickcapture_sensvalues(ard, baseline_remove=False)
    data = np.max(data, 0) - np.min(data, 0)
    return np.abs(data)


def get_sensmag(ard, baseline=[]):
    t, data = quickcapture_sensvalues(ard, baseline_remove=False)
    avgs = np.mean(data, 0)
    if len(baseline) < 4:
        return avgs
    else:
        return np.abs(avgs - baseline)


def get_sensfourier(times, data):
    '''
    returns the fourier spectrum for the sensor data
    :param times:
    :param data:
    :return:
    '''
    signals = data.shape[1]
    no_points = 256
    d = np.zeros([no_points, signals])
    t_new = np.linspace(times[0], times[-1], no_points)

    for i in range(signals):
        d_b = data[:, i]
        f = interpolate.interp1d(times, d_b)
        d_new = f(t_new)
        d[:, i] = d_new
        # plt.plot(t_new,d_new)
        # plt.plot(times, d_b)
        # plt.show()

    length_t = t_new[-1] - times[0]

    fs = no_points / length_t
    f = np.linspace(0, 1, -1 + no_points / 2) * fs / 2
    specs = np.zeros([no_points / 2 - 1, signals])

    for i in range(signals):
        Y = np.fft.fft(d[:, i], no_points) / no_points
        specs[:, i] = 2 * abs(Y[0:no_points / 2 - 1])
    return f, specs


def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i + 2], 16) for i in range(1, 6, 2)]


def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#" + "".join(["0{0:x}".format(v) if v < 16 else
                          "{0:x}".format(v) for v in RGB])


def color_dict(gradient):
    ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
    return {"hex": [RGB_to_hex(RGB) for RGB in gradient],
            "r": [RGB[0] for RGB in gradient],
            "g": [RGB[1] for RGB in gradient],
            "b": [RGB[2] for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j]))
            for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)
    return color_dict(RGB_list)


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


def rgb_gradient(color_dict):
    rgb_gradient = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    r_list = list(color_dict.values())[1]
    g_list = list(color_dict.values())[2]
    b_list = list(color_dict.values())[3]
    for i in range(0, 13):
        rgb_gradient[i] = [r_list[i], g_list[i], b_list[i]]
    return rgb_gradient


def charge_sensing_modev2(sensard, dispard, led_info, baseline):
    # setting up black background
    hexbg = image2ring(led_info, 'black.jpg')
    # this is the background string
    bgstr = compile_led_str(hexbg)
    # sending the background
    send_imgdata(bgstr, dispard)
    # Set up marquee
    process_list = get_imlist('ChargeAni')
    process_list = sorted(process_list, key=numericalSort)
    ##    #run marquee
    ##    print('Running CHARGE marquee')
    ##    for infile in process_list:
    ##        try:
    ##            led_info = image2ring(led_info, infile)
    ##            cw0 = gen_imgdata(led_info, infile)
    ##            t_delay = 0.001
    ##            send_imgdata(cw0,dispard)
    ##            time.sleep(t_delay)
    ##        except IndexError:
    ##            print ("this image is busted", infile)
    # setting up the background of a hexagon
    hexbg = image2ring(led_info, 'Hexagon.jpg')
    # this is the background string
    bgstr = compile_led_str(hexbg)
    # sending the background
    send_imgdata(bgstr, dispard)

    sensors = [sensor4, sensor5, sensor3, sensor2, sensor1]

    # led_info = led_reset(led_info)
    print('Beginning charge sensing mode.')
    while True:
        try:
            amps = np.abs(get_sensmag(sensard, baseline))
            # print(amps)
            qdisplay = amps > 20
            # print(qdisplay)
            # qdisplay is a list [t f t f t]
            # create random walk
            number = 1000
            [x, y, led_travel] = rand_walk(number, qdisplay)
            # print (led_travel.size)
            # print (x.size)
            # copy led lighting data of background
            sensactivate = copy.deepcopy(hexbg)
            # check sensors to light up those areas if on
            for i in range(5):
                if qdisplay[i]:
                    sensactivate = set_led_list(sensactivate, sensors[i], [255, 0, 0])
            # light up string of electrons
            sensactivate = set_led_list(sensactivate, led_travel, [0, 0, 255])
            # compile and send
            finalstr = compile_led_str(sensactivate)
            send_imgdata(finalstr, dispard)
        except KeyboardInterrupt:
            print(run)
            run = False


def amplitude_sensing_modev2(sensard, dispard, led_info, baseline):
    # setting up black background
    hexbg = image2ring(led_info, 'black.jpg')
    # this is the background string
    bgstr = compile_led_str(hexbg)
    # sending the background
    send_imgdata(bgstr, dispard)
    # Set up marquee
    process_list = get_imlist('MassAni')
    process_list = sorted(process_list, key=numericalSort)
    ##    #run marquee
    ##    print('Running Mass marquee')
    ##    for infile in process_list:
    ##        try:
    ##            led_info = image2ring(led_info, infile)
    ##            cw0 = gen_imgdata(led_info, infile)
    ##            t_delay = 0.001
    ##            send_imgdata(cw0,dispard)
    ##            time.sleep(t_delay)
    ##        except IndexError:
    ##            print ("this image is busted", infile)
    # setting up the background of a hexagon
    hexbg = image2ring(led_info, 'Hexagon.jpg')
    # initialize variable
    amplitude_color = [0, 0, 0]
    # this is the background string
    bgstr = compile_led_str(hexbg)
    # sending the background
    send_imgdata(bgstr, dispard)
    # set up plot
    x = np.linspace(0, 50, 50)
    xnew = np.linspace(0, 50, 500)
    y = np.zeros(50)
    run = True
    plt.ion()
    plt.axis([0, 50, 0, 100])
    print('Now sensing vibrations')
    # loop all of this
    while run:
        try:
            data = get_sensamp(sensard, baseline)
            # get gradient based on this data
            normalized_amp = data[2] / 70
            redness_amp = normalized_amp * 255
            blueness_amp = 255 - normalized_amp * 255
            amplitude_color = RGB_to_hex([redness_amp, 0, blueness_amp])
            color_dict = linear_gradient(amplitude_color, "#0000FF", 13)
            # print(color_dict)
            rgb = rgb_gradient(color_dict)
            print(rgb[0])
            # plot this stuff
            y = np.roll(y, -1)
            y[-1] = data[2]
            f = interpolate.interp1d(x, y, kind='cubic')
            ynew = f(xnew)
            plt.cla()
            plt.plot(xnew, ynew, 'k')
            axes = plt.gca()
            axes.set_ylim([0, 70])
            plt.pause(1e-3)
            # copy led lighting data of background
            sensactivate = copy.deepcopy(hexbg)
            # check sensors to light up those areas if on
            for led in led_info:
                if led['number'] in ring_1:
                    sensactivate = set_led_list(sensactivate, ring_1, rgb[0])
                elif led['number'] in ring_2:
                    sensactivate = set_led_list(sensactivate, ring_2, rgb[1])
                elif led['number'] in ring_3:
                    sensactivate = set_led_list(sensactivate, ring_3, rgb[2])
                elif led['number'] in ring_4:
                    sensactivate = set_led_list(sensactivate, ring_4, rgb[3])
                elif led['number'] in ring_5:
                    sensactivate = set_led_list(sensactivate, ring_5, rgb[4])
                elif led['number'] in ring_6:
                    sensactivate = set_led_list(sensactivate, ring_6, rgb[5])
                elif led['number'] in ring_7:
                    sensactivate = set_led_list(sensactivate, ring_7, rgb[6])
                elif led['number'] in ring_8:
                    sensactivate = set_led_list(sensactivate, ring_8, rgb[7])
                elif led['number'] in ring_9:
                    sensactivate = set_led_list(sensactivate, ring_9, rgb[8])
                elif led['number'] in ring_10:
                    sensactivate = set_led_list(sensactivate, ring_10, rgb[9])
                elif led['number'] in ring_11:
                    sensactivate = set_led_list(sensactivate, ring_11, rgb[10])
                elif led['number'] in ring_12:
                    sensactivate = set_led_list(sensactivate, ring_12, rgb[11])
                elif led['number'] in ring_13:
                    sensactivate = set_led_list(sensactivate, ring_13, rgb[12])
            # compile and send
            finalstr = compile_led_str(sensactivate)
            send_imgdata(finalstr, dispard)

        except KeyboardInterrupt:
            print(run)
            run = False


def magnitude_sensing_modev2(sensard, dispard, led_info, baseline):
    x = np.linspace(0, 50, 50)
    xnew = np.linspace(0, 50, 500)
    y = np.zeros(50)
    run = True
    plt.ion()
    while run:
        try:
            data = get_sensmag(sensard, baseline)
            y = np.roll(y, -1)
            y[-1] = np.mean(data)
            ynew = f(xnew)
            plt.cla()
            plt.plot(xnew, ynew, 'k')

            axes = plt.gca()
            # axes.set_ylim([0, 20])
            plt.pause(1e-3)

        except KeyboardInterrupt:
            print(run)
            run = False


def compile_led_str(led_info):
    string = ''
    for led in led_info:
        string += str(int(led['255color'][0])) + ','
        string += str(int(led['255color'][1])) + ','
        string += str(int(led['255color'][2])) + ','
    return string.encode()


def LED_sweep(led_info, dispard):
    blank = compile_led_str(led_info)
    send_imgdata(blank, dispard)
    time.sleep(0.5)
    for led in led_info:
        print(led['number'])
        led['255color'] = [255, 255, 255]
        string = compile_led_str(led_info)
        send_imgdata(string, dispard)
        led['255color'] = [0, 0, 0]
        ui = input('')


if __name__ == '__main__':

    # Setting up arduinos and waiting for connection
    sensard = serial.Serial('COM3', 500000, timeout=0.15)
    dispard = serial.Serial('COM5', 500000, timeout=4)
    time.sleep(3)

    # setting up led ring info
    led_info = make_led_coordinates(ring_radii, no_leds)

    # getting the baseline of the sensors for empty drum
    baseline = get_sensmag(sensard)
    print('The baseline is')
    print(baseline)
    # LED_sweep(led_info,dispard)
    while True:
        ui = input('What mode would you like - charge or mass?')
        if ui == 'charge':
            try:
                charge_sensing_modev2(sensard, dispard, led_info, baseline)
            except:
                print('Unexpected error:', sys.exc_info())
                pass
        elif ui == 'mass':
            try:
                amplitude_sensing_modev2(sensard, dispard, led_info, baseline)
            except:
                print('Unexpected error:', sys.exc_info())
                pass
        elif ui == 'magnitude':
            try:
                magnitude_sensing_modev2(sensard, dispard, led_info, baseline)
            except:
                print('Unexpected error:', sys.exc_info())
                pass

    # charge_sensing_mode(sensard,dispard,led_info,baseline)
    # amplitude_sensing_mode(sensard,dispard,led_info,baseline)
    # magnitude_sensing_mode(sensard,dispard,led_info,baseline)
