import serial
import time
from ring_init import *
from sensing_modes import *
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #Setting up arduinos and waiting for connection
    sensard = 0#serial.Serial('COM3', 500000, timeout=0.15)
    dispard = 0#serial.Serial('COM5', 500000, timeout=4)
    time.sleep(1)

    #setting up led ring info
    led_info = make_led_coordinates(ring_radii, no_leds)

    #getting the baseline of the sensors for empty drum
    baseline = 0#get_sensmag(sensard)
    print('The baseline is')
    print(baseline)
    #LED_sweep(led_info,dispard)
    while True:
        ui = input('What mode would you like - charge or mass?')
        if ui == 'charge':
            try:
                charge_sensing_modev2(sensard, dispard, led_info, baseline)
            except:
                print ('Unexpected error:', sys.exc_info())
                pass
        elif ui == 'mass':
            try:
                amplitude_sensing_modev2(sensard, dispard, led_info, baseline)
            except:
                print ('Unexpected error:', sys.exc_info())
                pass
        elif ui == 'magnitude':
            try:
                magnitude_sensing_modev2(sensard, dispard, led_info, baseline)
            except:
                print ('Unexpected error:', sys.exc_info())
                pass


    #charge_sensing_mode(sensard,dispard,led_info,baseline)
    #amplitude_sensing_mode(sensard,dispard,led_info,baseline)
    #magnitude_sensing_mode(sensard,dispard,led_info,baseline)

