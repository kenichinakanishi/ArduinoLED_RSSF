import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from PIL import ImageOps as imo
import time

#Get Files from folder
def get_imlist(path):
  """  Returns a list of filenames for
    all jpg images in a directory. """
  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpeg')]

#Sort files into numerical order
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def image2ring(led_info,imgfn,box_rad=1):
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
    img = imo.fit(img,(488,488),method=im.ANTIALIAS)
    img = img.rotate(-30)
    imarray = np.asarray(img)
    for led in led_info:
        ledx = led['x']
        ledy = led['y']
        color = np.average(
            np.average(
                imarray[int(ledx)-box_rad-1:int(ledx)+box_rad, int(ledy)-box_rad-1:int(ledy)+box_rad, :] / 255,
                0),
            0)[0:3]
        led['color'] = color
        led['255color'] = color * 255
    return led_info

def gen_imgdata(led_info,imgfn):
    '''
    This function generates the string that gets sent serially to the arduino to display an image.
    :param led_info:
    :return:
    '''
    led_info = image2ring(led_info,imgfn)
    string = ''
    for led in led_info:
        string+=str(int(led['255color'][0])) +','
        string+=str(int(led['255color'][1])) +','
        string+=str(int(led['255color'][2])) +','
    return string.encode()

def generate_video(fns,led_info):
    '''
    takes a list of file names and compiles them into a list of images to be uploaded sequentially to the arduino using
    upload_video()
    :param fns:
    :param led_info:
    :return:
    '''
    videolist = []
    for fn in fns:
        videolist.append(gen_imgdata(led_info,fn))

    return  videolist

def upload_video(imglist,arduino,loops=1,frame_delay=0.05):
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
            send_imgdata(img,arduino)
            time.sleep(frame_delay)