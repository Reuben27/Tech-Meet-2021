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

def blurring(img):
    image = Image.open(img)
    img_array = np.array(image)    
    rows,cols, temp = img_array.shape
    Blurred=cv2.blur(img_array,(BlurAmount,BlurAmount))
    st.image(Blurred)

def brightness(img,BrightnessValue):
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

st.title('Inter IIT Tech Meet 2021')

uploaded_files = st.file_uploader("Upload images", accept_multiple_files = True)
percent_testing = st.slider('Select % of images for training', 0, 100, step=25)

rotating_degree = st.slider('Select rotation degree', -45, 45, step=1, value=0)
rotater = st.button('Rotate')
if(rotater):
    for uploaded_file in uploaded_files:
        rotation(uploaded_file, rotating_degree)
    
