from plot_display import *
from color_gradients import *
from arduino_polling import *
import matplotlib.pyplot as plt
import numpy as np
from show_images import *
import time
import copy
import scipy.interpolate as interpolate

def compile_led_str(led_info):
    string = ''
    for led in led_info:
        string+=str(int(led['255color'][0])) +','
        string+=str(int(led['255color'][1])) +','
        string+=str(int(led['255color'][2])) +','
    return string.encode()

def LED_sweep(led_info,dispard):
    blank = compile_led_str(led_info)
    send_imgdata(blank,dispard)
    time.sleep(0.5)
    for led in led_info:
        print(led['number'])
        led['255color'] = [255,255,255]
        string = compile_led_str(led_info)
        send_imgdata(string,dispard)
        led['255color'] = [0, 0, 0]
        ui = input('')

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
    import numpy as np
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


def magnitude_sensing_modev2(sensard,dispard,led_info,baseline):
    x = np.linspace(0,50,50)
    xnew = np.linspace(0,50,500)
    y = np.zeros(50)
    run = True
    plt.ion()
    while run:
        try:
            data = get_sensmag(sensard, baseline)
            y = np.roll(y,-1)
            y[-1] = np.mean(data)
            ynew = f(xnew)
            plt.cla()
            plt.plot(xnew,ynew, 'k')

            axes = plt.gca()
            #axes.set_ylim([0, 20])
            plt.pause(1e-3)

        except KeyboardInterrupt:
            print(run)
            run=False
