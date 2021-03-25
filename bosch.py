import streamlit as st
import numpy as np
import cv2 as cv2 
from PIL import Image

###### Image Augmentation Options ######
def rotation(img_array, degree):   
    rows, cols, temp = img_array.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    dst = cv2.warpAffine(img_array,M,(cols,rows))
    st.image(dst)

def translation(img_array,x,y):   
    rows, cols, temp = img_array.shape
    M = np.float32([[1,0,x],[0,1,-y]])
    dst = cv2.warpAffine(img_array,M,(cols,rows))
    st.image(dst)

def blurring(img_array,BlurAmount):  
    rows, cols, temp = img_array.shape
    Blurred = cv2.blur(img_array,(BlurAmount,BlurAmount))
    st.image(Blurred)

def brighter(img,factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,factor)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #enhancer = ImageEnhance.Brightness(image)
    #im_output = enhancer.enhance(factor)
    st.image(img)

def flip(img_array):   
    rows, cols, temp = img_array.shape
    flip=cv2.flip(img_array,1)
    st.image(flip)

def all_changes(img_array, degree, x, y, brighter, BlurAmount, genre):
    rows,cols, temp = img_array.shape
    #Rotation
    if (degree != None):
        rows, cols, temp = img_array.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
        img_array = cv2.warpAffine(img_array,M,(cols,rows))

    #Translation
    if (x != None and y != None):
        rows, cols, temp = img_array.shape
        M = np.float32([[1,0,x],[0,1,-y]])
        img_array = cv2.warpAffine(img_array,M,(cols,rows))

    #Brightness
    if (brighter != None):
        hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v,brighter)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img_array = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    #Blurred
    if (BlurAmount != None):
        img_array = cv2.blur(img_array,(BlurAmount,BlurAmount))

    #Flip
    if (genre == "Yes"):
        rows,cols, temp = img_array.shape
        img_array=cv2.flip(img_array,1)

    st.image(img_array)

###### Image Augmentation Options ######

preview_image = Image.open("meteor.png")
original_img_array = np.array(preview_image)
#st.image(original_img_array)
modified_img_array = np.array(preview_image)
rotating_degree = None 
x_direction = None 
y_direction = None
blur = None 

st.sidebar.title("Bosch Traffic Sign Recognition")
st.sidebar.subheader("With the advancements in AI and the development of computing capabilities in the 21st century, millions of processes around the globe are being automated like never before. The automobile industry is transforming,")

#uploaded_file = Image.open("iitgnlogo.png")
#original_img_array = np.array(uploaded_file)
#st.image(original_img_array)

st.title('Inter IIT Tech Meet 2021')
st.header("Image Augmentation Options")

st.subheader("Training/Testing")
training_percent = st.slider('Select % of images for training', 20, 80, step = 20)

st.subheader("Rotate Images")
rotating_degree = st.slider('Select the degree upto which you want to rotate the image', -20, 20, step = 1, value = 0)

st.subheader("Flip Images")
flip_or_not = st.radio("Do you want to flip the images?", ('Yes', 'No'), 1)

st.subheader("Brighten Images")
brightness_factor = st.slider("Select the brightness factor", 0.75, 1.25, step = 0.05, value = 1.00)

st.subheader("Translate Images")
col_x, col_y = st.beta_columns(2)
with col_x:
    x_direction = st.slider("Select the translation along x direction", -0.2, 0.2, step = 0.02, value = 0.0)
with col_y:
    y_direction = st.slider("Select the translation along y direction", -0.2, 0.2, step = 0.02, value = 0.0)

Augment_Image = st.button("Augment Image")

if Augment_Image:
    st.write(training_percent)
    st.write(rotating_degree)
    st.write(flip_or_not)
    st.write(brightness_factor)
    st.write(x_direction)
    st.write(y_direction)
    all_changes(modified_img_array, rotating_degree, x_direction, y_direction, brightness_factor, blur, flip_or_not)