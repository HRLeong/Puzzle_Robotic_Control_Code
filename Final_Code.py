#Import libraries and dependencies
import serial
import torch
torch.cuda.is_available()
from matplotlib import pyplot as plt 
import numpy as np
import cv2
import pandas as pd
import os
from pydexarm import Dexarm
import time,sys
arduino_serial = serial.Serial('com4', 115200)
dexarm = Dexarm(port="COM3")

#Set the correct directory in which the programme is to be run

os.chdir('C:/Users/feynm/Desktop/yolov5_puzzle_detection/yolov5')

#Load the model and weights. Set up the Classification/Localisation and Orientation functions

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'runs/train/exp2/weights/bests13.pt', force_reload=True)
model_orientation = torch.hub.load('ultralytics/yolov5', 'custom', path = 'runs/train/exp2/weights/best_Pos_orien.pt', force_reload=True)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1280
height = 720
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def capture_scene():
    while capture.isOpened():
        ret, frame = capture.read()

        result = model(frame)
        dataframe = result.pandas().xyxy[0]



        cv2.imshow('Results', np.squeeze(result.render()))

        if dataframe[dataframe['class'] == 0]['name'].count() == 4 and dataframe[dataframe['class'] == 1]['name'].count() == 4 and dataframe[dataframe['class'] == 2]['name'].count() == 1 :
            cv2.imwrite('image' + '.jpg', frame)
            dataframe.to_csv('location_results.csv')
            break
        if cv2.waitKey(10) and 0xFF == ord('q'):
            break

      
    capture.release()
    cv2.destroyAllWindows()  

capture_2 = cv2.VideoCapture(3, cv2.CAP_DSHOW)
capture_2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width_1 = 640
height_1 = 640
capture_2.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
capture_2.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)

def orientation_check():
    _count = 0
    _started = False

    while capture_2.isOpened():
        ret, frame = capture_2.read()

        if _count < 5:
            _count += 1
            continue

        results = model_orientation(frame)
        dataframe_1 = results.pandas().xyxy[0]
        if not _started:
            arduino_serial.write("ROTATE".encode())
            _started = True
            print("rotation started")
            time.sleep(1.5)


        cv2.imshow('Results', np.squeeze(results.render()))

        if dataframe_1[dataframe_1['class'] == 0]['name'].count() == 1:
            arduino_serial.write("STOP".encode())
            print("rotation stopped")
            time.sleep(1.5)
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    
    cv2.destroyAllWindows()
    

    return True
            
# Compute the coordinates of the centre of the puzzle piece

def Puzzle_Coordinates():
    df = pd.read_csv('location_results.csv')
    Corner_df = df[df['class']==0].loc[:, 'xmin':'ymax']
    Side_df = df[df['class']==1].loc[:, 'xmin':'ymax']
    Centre_df = df[df['class']==2].loc[:, 'xmin':'ymax']

    global Corner_point_coordinates
    global Side_point_coordinates
    global Centre_point_coordinates

    Corner_coordinate = Corner_df.to_numpy()
    Side_coordinate = Side_df.to_numpy()
    Centre_coordinate = Centre_df.to_numpy()

    Corner_point_coordinates = np.zeros(shape=(Corner_coordinate.shape[0],2), dtype=int)

    for i in range(0,Corner_coordinate.shape[0]):
        x_centre = (Corner_coordinate[i,2] - Corner_coordinate[i,0])/2
        y_centre = (Corner_coordinate[i,3] - Corner_coordinate[i,1])/2

        y_real = 0.430 * (Corner_coordinate[i,1] + y_centre) + 115
        x_real = 0.430 * (Corner_coordinate[i,0] + x_centre) - 30

        Corner_point_coordinates[i] = [round(y_real),round(x_real)]
        
    
    Side_point_coordinates = np.zeros(shape=(Side_coordinate.shape[0],2), dtype=int)

    for i in range(0,Side_coordinate.shape[0]):
        x_centre = (Side_coordinate[i,2] - Side_coordinate[i,0])/2
        y_centre = (Side_coordinate[i,3] - Side_coordinate[i,1])/2

        y_real = 0.430 * (Side_coordinate[i,1] + y_centre) + 115
        x_real = 0.430 * (Side_coordinate[i,0] + x_centre) - 30

        Side_point_coordinates[i] = [round(y_real),round(x_real)]
    

    Centre_point_coordinates = np.zeros(shape=(Centre_coordinate.shape[0],2), dtype=int)

    for i in range(0,Centre_coordinate.shape[0]):
        x_centre = (Centre_coordinate[i,2] - Centre_coordinate[i,0])/2
        y_centre = (Centre_coordinate[i,3] - Centre_coordinate[i,1])/2

        y_real = 0.430 * (Centre_coordinate[i,1] + y_centre) + 115
        x_real = 0.430 * (Centre_coordinate[i,0] + x_centre) - 30

        Centre_point_coordinates[i] = [round(y_real),round(x_real)]

    


#Position and move the puzzle piece to its correct orientation and final location 


def move_to_stepper_and_drop():
    dexarm.move_to([0,370,35,645])
    time.sleep(3)
    dexarm.move_to([0,370,18,645])
    time.sleep(5)
    arduino_serial.write("SOL_CLOSE".encode())
    time.sleep(2)
    dexarm.move_to([0,370,35,645])
    time.sleep(2)
    dexarm.move_to([0,220,35,645])
    time.sleep(2)
    return True

def move_to_stepper_and_pick_up():
    dexarm.move_to([0,370,35,645])
    time.sleep(3)
    dexarm.move_to([0,370,18,645])
    time.sleep(3)
    arduino_serial.write("SOL_OPEN".encode())
    time.sleep(2)
    dexarm.move_to([0,370,35,645])
    time.sleep(2)
    return True

def move_to_side_location():
    dexarm.move_to([0,220,30,585])
    time.sleep(3)
    dexarm.move_to([0,220,-25,585])
    time.sleep(3)
    dexarm.move_to([0,220,-37,585])
    time.sleep(3)
    arduino_serial.write("SOL_CLOSE".encode())
    time.sleep(2)
    dexarm.move_to([0,220,-20,585])
    time.sleep(2)
    return True

def move_to_corner_location():
    dexarm.move_to([0,220,30,510])
    time.sleep(3)
    dexarm.move_to([0,220,-25,510])
    time.sleep(3)
    dexarm.move_to([0,220,-37,510])
    time.sleep(3)
    arduino_serial.write("SOL_CLOSE".encode())
    time.sleep(2)
    dexarm.move_to([0,220,-20,510])
    time.sleep(2)
    return True

def move_to_centre_location():
    dexarm.move_to([0,220,30,670])
    time.sleep(3)
    dexarm.move_to([0,220,-25,670])
    time.sleep(3)
    dexarm.move_to([0,220,-37,670])
    time.sleep(3)
    arduino_serial.write("SOL_CLOSE".encode())
    time.sleep(2)
    dexarm.move_to([0,220,-20,670])
    time.sleep(2)
    return True

def go_home():
    dexarm.move_to([0,300,0,0])
    return True

def move_centre(coordinate):
    for i in range(0,coordinate.shape[0]):
        dexarm.move_to([0,coordinate[i,0],-25,coordinate[i,1]])
        time.sleep(5)
        dexarm.move_to([0,coordinate[i,0],-37,coordinate[i,1]])
        time.sleep(2)
        arduino_serial.write("SOL_OPEN".encode())
        time.sleep(1.5)
        dexarm.move_to([0,coordinate[i,0],-25,coordinate[i,1]])
        time.sleep(5)
        move_to_stepper_and_drop()
        orientation_check()
        move_to_stepper_and_pick_up()
        move_to_centre_location()
        

def move_side(coordinate):
    for i in range(0,coordinate.shape[0]):
        dexarm.move_to([0,coordinate[i,0],-25,coordinate[i,1]])
        time.sleep(5)
        dexarm.move_to([0,coordinate[i,0],-37,coordinate[i,1]])
        time.sleep(2)
        arduino_serial.write("SOL_OPEN".encode())
        time.sleep(1.5)
        dexarm.move_to([0,coordinate[i,0],-25,coordinate[i,1]])
        time.sleep(5)
        move_to_stepper_and_drop()
        orientation_check()
        move_to_stepper_and_pick_up()
        move_to_side_location()
        

def move_corner(coordinate):
    for i in range(0,coordinate.shape[0]):
        dexarm.move_to([0,coordinate[i,0],-25,coordinate[i,1]])
        time.sleep(5)
        dexarm.move_to([0,coordinate[i,0],-37,coordinate[i,1]])
        time.sleep(2)
        arduino_serial.write("SOL_OPEN".encode())
        time.sleep(1.5)
        dexarm.move_to([0,coordinate[i,0],-25,coordinate[i,1]])
        time.sleep(5)
        move_to_stepper_and_drop()
        print("here1")
        orientation_check()
        print("here2")
        move_to_stepper_and_pick_up()
        move_to_corner_location()
        

capture_scene()
Puzzle_Coordinates()
move_centre(Centre_point_coordinates)
move_corner(Corner_point_coordinates)
move_side(Side_point_coordinates)
go_home()

