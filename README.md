# Puzzle Classification and Localisation ML Model with Rotrics & Arduino Integration

This code base is used in the PULSAR project for ESP3902 Major Design Project I at the National University of Singapore. This includes the code for running the YOLOv5 model as well as the library to inteface with the Rotrics DexArm as well as the Arduino Mega 2560

![This is the project setup](https://user-images.githubusercontent.com/95577932/202489303-bc66bcc2-84f1-4c74-88e2-96a8cd133a6c.png)

## System Requirements 
### Python libraries and environment
This project uses Python 3.10.8 in Visual Studio Code with the following libraries and dependencies:
* [serial](https://pypi.python.org/pypi/pyserial)
* [PyTorch](https://github.com/pytorch/pytorch#from-source)
* [matplotlib](https://matplotlib.org/stable/users/installing/index.html)
* [numpy](https://numpy.org/install/)
* [OpenCv](https://pypi.org/project/opencv-python/)
* [pandas](https://pypi.org/project/pandas/)
### YOLOv5 Computer Vision Model
Computer Vision model utilises YOLOv5 by Ultralytics to train and deploy the model. The Repository can be cloned to a local machine or a cloud-based IDE for faster training
```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```
The reference tutorial used to get started with YOLOv5 can be found [here](https://www.youtube.com/watch?v=tFNJGim3FXw&t=994s)
### Rotrics DexArm API Library
Python Communication with the Rotrics DexArm System requires downloading the API
* [pydexarm](https://github.com/Rotrics-Dev/DexArm_API)


## Contributors
The code base for this repository is contributed by the following members:
* Leong Han Ren (National University of Singapore)
* Lee Eun Seok (University of Waterloo)

## License

This code base is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License
