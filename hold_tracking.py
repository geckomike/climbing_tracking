#Written by Mike Richardson 11/10/2021

#This script combines the pre-trained object tracking model with the
#arduino output. The object tracking is based on the yolov5 ultralytrics
#model.

#The first part of the code was developed using a tutorial from
#Nicholas Renotte and can be found here: https://www.youtube.com/watch?v=tFNJGim3FXw&t=1063s
#The second part of the code sends serial commands to an Arduino Uno and the
#Cthulhu Shield, details can be found here: https://github.com/SapienLLCdev/Cthulhu

#The Cthulhu Shield takes hex key commands to turn on each individual
#electrode arranged in a 4x4 matrix (there are 2 additional electrodes
#on the tongue display which are ignored here for simplicity).
#The elecrodes are arranged as follows, with each electrodes serial
#code.

#To turn on a specific electrode, serial write the electrode's
#hexidecimal number


#                           (tip of tongue)

#                [['0x03', '0x04', '0x05', '0x06'],
#  (left side)    ['0x07', '0x08', '0x09', '0x0a'],      (right side)
#                 ['0x0b', '0x0c', '0x0d', '0x0e'],
#                 ['0x0f', '0x10', '0x11', '0x12']]

#                           (back of tongue)

##############################################################

#Set up libraries and functions______________________________

#Import libraries
import os
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
import serial
import time



#Load in pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                        path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)

colour = 'spots'

def start_up():
    print('')
    print('Written by Mike Richardson 11/10/2021.')
    print('This programme is designed to work with the Cthulhu Shield Sensory Substitution Device')
    print('and it uses ultralytics yolov5 object detection to track climbing holds')
    print('')
    print('To test whether the model loaded correctly:')
    print('press i to perform test_frame()     to test a single example frame')
    print('press v to perform test_video()     to test a sample video clip')
    print('')
    print('To track holds in real time run the function:')
    print('press l to run live_tracking()')
    print('or q to terminate the programme')
    print('################################################################')

    command = input('Please select an option... ')

    if command == 'q':
        print('End of programme')
    elif command == 'i':
        print('Testing model on sample image')
        test_frame()
    elif command == 'v':
        print('Testing model on sample video')
        test_video()
        wait_command()
    elif command == 'l':
        print('Running live hold detection')
        live_tracking()
        wait_command()
    else:
        pass

def wait_command():
    command = input('Please select an option... ')

    if command == 'q':
        print('End of programme')
    elif command == 'i':
        print('Testing model on sample image')
        test_frame()
    elif command == 'v':
        print('Testing model on sample video')
        test_video()
        wait_command()
    elif command == 'l':
        print('Running live hold detection')
        live_tracking()
        wait_command()
    else:
        pass

#Store each column of electrodes in a variable to
col_1 = [0x03, 0x07, 0x0b, 0x0f, 0x00]
col_1_2 = [0x03, 0x07, 0x0b, 0x0f, 0x04, 0x08, 0x0c, 0x10, 0x00]
col_1_2_3 = [0x03, 0x07, 0x0b, 0x0f, 0x04, 0x08, 0x0c, 0x10, 0x05, 0x09, 0x0d, 0x11, 0x00]
col_1_2_3_3 = [0x03, 0x07, 0x0b, 0x0f, 0x04, 0x08, 0x0c, 0x10, 0x05, 0x09, 0x0d, 0x11,
                0x06, 0x0a, 0x0e, 0x12, 0x00]

col_2 = [0x04, 0x08, 0x0c, 0x10, 0x00]
col_2_3 = [0x04, 0x08, 0x0c, 0x10, 0x05, 0x09, 0x0d, 0x11, 0x00]
col_2_3_4 = [0x04, 0x08, 0x0c, 0x10, 0x05, 0x09, 0x0d, 0x11, 0x06, 0x0a, 0x0e, 0x12, 0x00]

col_3 = [0x05, 0x09, 0x0d, 0x11, 0x00]
col_3_4 = [0x05, 0x09, 0x0d, 0x11, 0x06, 0x0a, 0x0e, 0x12, 0x00]

col_4 = [0x06, 0x0a, 0x0e, 0x12, 0x00]

stop = [0x00, 0x00]

#This function will update the Cthulhu Shield with hold locations
#It is called inside the live_tracking() function
def update_display(row, colour):

    if row.str.contains(colour).any():

        if row.xmin == 100.0 or 200.0 and row.xmax == 100.0 or 200.0:
            ser.write(serial.to_bytes(ccol_1))
        elif row.xmin == 100.0 or 200.0 and row.xmax == 300.0 or 400.0:
            ser.write(serial.to_bytes(col_1_2))
        elif row.xmin == 100.0 or 200.0 and row.xmax == 500.0 or 600.0:
            ser.write(serial.to_bytes(col_1_2_3))
        elif row.xmin == 100.0 or 200.0 and row.xmax == 700.0 or 800.0:
            ser.write(serial.to_bytes(col_1_2_3_4))

        elif row.xmin == 300.0 or 400.0 and row.xmax == 300.0 or 400.0:
            ser.write(serial.to_bytes(col_2))
        elif row.xmin == 300.0 or 400.0 and row.xmax == 500.0 or 600.0:
            ser.write(serial.to_bytes(col_2_3))
        elif row.xmin == 300.0 or 400.0 and row.xmax == 700.0 or 800.0:
            ser.write(serial.to_bytes(col_2_3_4))

        elif row.xmin == 500.0 or 600.0 and row.xmax == 500.0 or 600.0:
            ser.write(serial.to_bytes(col_3))
        elif row.xmin == 500.0 or 600.0 and row.xmax == 700.0 or 800.0:
            ser.write(serial.to_bytes(col_3_4))

        elif row.xmin == 700.0 or 800.0 and row.xmax == 700.0 or 800.0:
            ser.write(serial.to_bytes(col_4))

        else:
            ser.write(serial.to_bytes(stop))

    else:
        ser.write(serial.to_bytes(stop))


#This function emulates the update_display function, but prints to the console
#rather than via serial. It can be used for testing without connecting an Arduino.
def update_display_emulator(row, colour):

    if row.str.contains(colour).any():

        if row.xmin == 100.0 or 200.0 and row.xmax == 100.0 or 200.0:
            print(col_1)
        elif row.xmin == 100.0 or 200.0 and row.xmax == 300.0 or 400.0:
            print(col_1_2)
        elif row.xmin == 100.0 or 200.0 and row.xmax == 500.0 or 600.0:
            print(col_1_2_3)
        elif row.xmin == 100.0 or 200.0 and row.xmax == 700.0 or 800.0:
            print(col_1_2_3_4)

        elif row.xmin == 300.0 or 400.0 and row.xmax == 300.0 or 400.0:
            print(col_2)
        elif row.xmin == 300.0 or 400.0 and row.xmax == 500.0 or 600.0:
            print(col_2_3)
        elif row.xmin == 300.0 or 400.0 and row.xmax == 700.0 or 800.0:
            print(col_2_3_4)

        elif row.xmin == 500.0 or 600.0 and row.xmax == 500.0 or 600.0:
            print(col_3)
        elif row.xmin == 500.0 or 600.0 and row.xmax == 700.0 or 800.0:
            print(col_3_4)

        elif row.xmin == 700.0 or 800.0 and row.xmax == 700.0 or 800.0:
            print(col_4)

        else:
            print(stop)

    else:
        print(stop)


################################################################

#Test the model on pre-recorded data files______________________

#Test model on a single frame
def test_frame():
    img = os.path.join('data', 'images', 'image9ea506a8-577c-11ec-aecb-1e00a2015b79.jpg')

    results = model(img)
    results.print()
    wait_command()


#Function to choose which colour to track.
def pick_colour():
    global colour
    print('Choose colour to track:')
    print('Options: red, green, yellow, spots, orange')
    colour = input('Type colour: ')

    return colour


#Test model on a video file
#This function does not require the Cthulhu Shield to be connected
#It will run the model on a video file and display in a window
def test_video():
    cap = cv2.VideoCapture('data/hold_footage_2.mov')

    pick_colour()

    while cap.isOpened():
        ret, frame = cap.read()

        #make detections
        results = model(frame)

        cv2.imshow('Climbing Tracking', np.squeeze(results.render()))
        object_array = results.pandas().xyxy[0] #takes coordinates from frame
        rounded_array = np.around(object_array, -2) #simplifes pixel density


        for index, row in rounded_array.iterrows():
            update_display_emulator(row, colour)


        if cv2.waitKey(10) & 0xFF == ord('q'): #hold q to kill window
            break

    cap.release()
    cv2.destroyAllWindows()
    for i in range(1,5):  #This is to handle a bug cv2 has
        cv2.waitKey(1)

###############################################################

#The hold detection function
#The Cthulhu Shield must be connected for this to work.
def live_tracking():
    cap = cv2.VideoCapture(1) #this may change depending on webcam

    pick_colour()

    while cap.isOpened():
        ret, frame = cap.read()

        #make detections
        results = model(frame[490:590, 560:1360]) #this crops frame

        cv2.imshow('Climbing Tracking', np.squeeze(results.render()))
        object_array = results.pandas().xyxy[0] #takes coordinates from frame
        rounded_array = np.around(object_array, -2) #simplifes pixel density

        #print(rounded_array)
        for index, row in rounded_array.iterrows():
            update_display_2(row, colour)



        if cv2.waitKey(10) & 0xFF == ord('q'): #hold q to kill window
            break

    cap.release()
    cv2.destroyAllWindows()
    for i in range(1,5):  #This is to handle a bug cv2 has
        cv2.waitKey(1)

##############################################################

#Open serial port. Edit port depending on serial port
#ser = serial.Serial(port='/dev/cu.usbmodem11401',baudrate=115200, timeout=5)


if __name__ == "__main__":
    start_up()
