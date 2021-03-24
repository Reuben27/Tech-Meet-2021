import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv2 
from PIL import Image

def rotation(img, degree):
    image = Image.open(img)
    img_array = np.array(image)    
    rows,cols, temp = img_array.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    dst = cv2.warpAffine(img_array,M,(cols,rows))
    st.image(dst)

def translation(img,x,y):
    image = Image.open(img)
    img_array = np.array(image)    
    rows,cols, temp = img_array.shape
    M = np.float32([[1,0,x],[0,1,-y]])
    dst = cv2.warpAffine(img_array,M,(cols,rows))
    st.image(dst)

def blurring(img, BlurAmount):
    image = Image.open(img)
    img_array = np.array(image)    
    rows,cols, temp = img_array.shape
    Blurred = cv2.blur(img_array,(BlurAmount,BlurAmount))
    st.image(Blurred)

def brightness(img,BrightnessValue):
    # Taking a lot of time
    image = Image.open(img)
    img_array = np.array(image)    
    rows,cols, temp = img_array.shape
    new_image = np.zeros(img_array.shape, img_array.dtype)
    if(-1<=BrightnessValue<=1):
        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                for c in range(img_array.shape[2]):
                    new_image[y,x,c] = np.clip(img_array[y,x,c] + BrightnessValue*255, 0, 255)
        st.image(new_image)
    else:
        print("enter value between -1 and 1")
                    
def zoom(img,x1,x2,y1,y2):
    # error: OpenCV(4.3.0) C:\projects\opencv-python\opencv\modules\imgproc\src\resize.cpp:3929: error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'
    # Traceback:
    # File "c:\users\reube\appdata\local\programs\python\python38\lib\site-packages\streamlit\script_runner.py", line 333, in _run_script
    #     exec(code, module.__dict__)
    # File "D:\01 - Projects\Machine Learning\Tech-Meet-2021\Streamlit\app.py", line 93, in <module>
    #     zoom(uploaded_file,x1,x2,y1,y2)
    # File "D:\01 - Projects\Machine Learning\Tech-Meet-2021\Streamlit\app.py", line 51, in zoom
    res = cv2.resize(crop,(28,28), interpolation = cv2.INTER_CUBIC)
    image = Image.open(img)
    img_array = np.array(image)    
    rows,cols, temp = img_array.shape
    if(0<=x1<=np.shape(img_array)[0] and 0<=x2<=np.shape(img_array)[0] and 0<=y1<=np.shape(img_array)[1] and 0<=y2<=np.shape(img_array)[1]):
        crop = img_array[y1:y2,x1:x2]
        res = cv2.resize(crop,(28,28), interpolation = cv2.INTER_CUBIC)
        st.image(res)
    else:
        print("enter x value less than {} and y value leass than {}".format(np.shape(img)[0],np.shape(img)[1]))

def flip(img):
    image = Image.open(img)
    img_array = np.array(image)    
    rows,cols, temp = img_array.shape
    flip=cv2.flip(img_array,1)
    st.image(flip)

st.sidebar.header("Image Augmentation Options")
#st.sidebar.subheader("Ishan Prayagi")
#st.sidebar.subheader("Reuben Devanesan")

st.title('Inter IIT Tech Meet 2021')

st.subheader("Upload images")
uploaded_files = st.file_uploader("", accept_multiple_files = True)

st.subheader("Training/Testing")
percent_testing = st.slider('Select % of images for training', 0, 100, step=25)

st.subheader("Rotate Images")
rotating_degree = st.slider('Select rotation degree', -45, 45, step=1, value=0)
rotater = st.button('Rotate')

if(rotater):
    for uploaded_file in uploaded_files:
        rotation(uploaded_file, rotating_degree)

st.subheader("Flip Images")
flipper = st.button('Flip')
if(flipper):
    for uploaded_file in uploaded_files:
        flip(uploaded_file)

st.subheader("Brighten Images")
bright = st.number_input("Enter brightness amount")
brighten = st.button('Brighten')
if(brighten):
    for uploaded_file in uploaded_files:
        brightness(uploaded_file, bright)

st.subheader("Blur Images")
blur = st.number_input("Enter blurring amount", format = "%d", value = 1)
blurrer = st.button('Blur')
if(blurrer):
    for uploaded_file in uploaded_files:
        blurring(uploaded_file, blur)

col1,col2 = st.beta_columns(2)
st.subheader("Translate Images")
with col1:
    x = st.number_input("x direction", format = "%d", value = 1)
with col2:
    y = st.number_input("y direction", format = "%d", value = 1)
translater = st.button('Translate')
if(translater):
    for uploaded_file in uploaded_files:
        translation(uploaded_file, x, y)

#col1,col2,col3,col4 = st.beta_columns(4)
# x1 = st.number_input("Enter x1", format = "%d", value = 1)
# x2 = st.number_input("Enter x2", format = "%d", value = 1)
# y1 = st.number_input("Enter y1", format = "%d", value = 1)
# y2 = st.number_input("Enter y2", format = "%d", value = 1)
# zoomer = st.button('Zoom')

# if(zoomer):
#     for uploaded_file in uploaded_files:
#         zoom(uploaded_file,x1,x2,y1,y2)