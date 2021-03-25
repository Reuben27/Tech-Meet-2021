from keras.layers import Flatten
from keras.layers import Activation
from keras.activations import relu
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.models import Sequential
import keras
from skimage import data, exposure
import os
import matplotlib.pyplot as plt
import cv2
import PIL
import tensorflow as tf
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
    M = np.float32([[1,0,x*img_array.shape[0]],[0,1,-y*img_array.shape[1]]])
    dst = cv2.warpAffine(img_array,M,(cols,rows))
    st.image(dst)

def blurring(img_array,BlurAmount):  
    rows, cols, temp = img_array.shape
    Blurred = cv2.blur(img_array,(BlurAmount,BlurAmount))
    st.image(Blurred)

def brighter(img,factor):
    if(factor <= 100):
        factor = factor - 100
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
        M = np.float32([[1,0,x*img_array.shape[0]],[0,1,-y*img_array.shape[1]]])
        img_array = cv2.warpAffine(img_array,M,(cols,rows))

    #Brightness
    if (brighter != None):
        if(brighter <= 100):
            brighter = brighter - 100
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

preview_image = Image.open("A.png")
original_img_array = np.array(preview_image)
#st.image(original_img_array)
modified_img_array = np.array(preview_image)
rotating_degree = None 
x_direction = None 
y_direction = None
blur = None 

st.sidebar.title("Bosch Traffic Sign Recognition")
#st.sidebar.subheader("With the advancements in AI and the development of computing capabilities in the 21st century, millions of processes around the globe are being automated like never before. The automobile industry is transforming,")

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
    # st.write(training_percent)
    # st.write(rotating_degree)
    # st.write(flip_or_not)
    # st.write(brightness_factor)
    # st.write(x_direction)
    # st.write(y_direction)
    col_1, col_2 = st.beta_columns(2)
    with col_1:
        st.subheader("Original Image")
        st.image(original_img_array)
    with col_2:
        st.subheader("Augmentated Image")
        all_changes(modified_img_array, rotating_degree, x_direction,
                    y_direction, int(brightness_factor*100), blur, flip_or_not)


def preprocessing_function(x):
    x = x/255.
    image = exposure.equalize_adapthist(x, clip_limit=0.1)
    return image




class Localization(tf.keras.layers.Layer):
    def __init__(self, f1, f2, f3, **kwargs):
        super(Localization, self).__init__()
        self.pool1 = tf.keras.layers.MaxPool2D(2)
        self.conv1 = tf.keras.layers.Conv2D(
            f1, 5, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D(2)
        self.conv2 = tf.keras.layers.Conv2D(
            f2, 5, activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPool2D(2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(f3, activation='relu')
        self.fc2 = tf.keras.layers.Dense(6, activation=None, bias_initializer=tf.keras.initializers.constant([
                                         1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), kernel_initializer='zeros')

    def build(self, input_shape):
        print("Building Localization Network with input shape:", input_shape)

    def compute_output_shape(self, input_shape):
        return [None, 6]

    def get_config(self):

        config = super(Localization, self).get_config().copy()
        config.update({
            'f1': self.f1,
            'f2': self.f2,
            'f3': self.f3
        })
        return config

    def call(self, inputs):
        x = self.pool1(inputs)
        x = self.conv1(x)
        x = self.pool2(x)
        x = self.conv2(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        theta = self.fc2(x)
        theta = tf.keras.layers.Reshape((2, 3))(theta)
        return theta

#Part of Spatial Transformer module
# implementation from https://towardsdatascience.com/implementing-spatial-transformer-network-stn-in-tensorflow-bf0dc5055cd5


class BilinearInterpolation(tf.keras.layers.Layer):
    def __init__(self, height=40, width=40):
        super(BilinearInterpolation, self).__init__()
        self.height = height
        self.width = width

    def compute_output_shape(self, input_shape):
        return [None, self.height, self.width, 1]

    def get_config(self):
        return {
            'height': self.height,
            'width': self.width,
        }

    def build(self, input_shape):
        print("Building Bilinear Interpolation Layer with input shape:", input_shape)

    def advance_indexing(self, inputs, x, y):
        '''
        Numpy like advance indexing is not supported in tensorflow, hence, this function is a hack around the same method
        '''
        shape = tf.shape(inputs)
        batch_size, _, _ = shape[0], shape[1], shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, self.height, self.width))
        indices = tf.stack([b, y, x], 3)
        return tf.gather_nd(inputs, indices)

    def call(self, inputs):
        images, theta = inputs
        homogenous_coordinates = self.grid_generator(batch=tf.shape(images)[0])
        return self.interpolate(images, homogenous_coordinates, theta)

    def grid_generator(self, batch):
        x = tf.linspace(-1, 1, self.width)
        y = tf.linspace(-1, 1, self.height)

        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))
        homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)])
        homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0)
        homogenous_coordinates = tf.tile(homogenous_coordinates, [batch, 1, 1])
        homogenous_coordinates = tf.cast(
            homogenous_coordinates, dtype=tf.float32)
        return homogenous_coordinates

    def interpolate(self, images, homogenous_coordinates, theta):

        with tf.name_scope("Transformation"):
            transformed = tf.matmul(theta, homogenous_coordinates)
            transformed = tf.transpose(transformed, perm=[0, 2, 1])
            transformed = tf.reshape(
                transformed, [-1, self.height, self.width, 2])

            x_transformed = transformed[:, :, :, 0]
            y_transformed = transformed[:, :, :, 1]

            x = ((x_transformed + 1.) * tf.cast(self.width, dtype=tf.float32)) * 0.5
            y = ((y_transformed + 1.) *
                 tf.cast(self.height, dtype=tf.float32)) * 0.5

        with tf.name_scope("VariableCasting"):
            x0 = tf.cast(tf.math.floor(x), dtype=tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.math.floor(y), dtype=tf.int32)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, 0, self.width-1)
            x1 = tf.clip_by_value(x1, 0, self.width-1)
            y0 = tf.clip_by_value(y0, 0, self.height-1)
            y1 = tf.clip_by_value(y1, 0, self.height-1)
            x = tf.clip_by_value(x, 0, tf.cast(
                self.width, dtype=tf.float32)-1.0)
            y = tf.clip_by_value(y, 0, tf.cast(
                self.height, dtype=tf.float32)-1)

        with tf.name_scope("AdvanceIndexing"):
            Ia = self.advance_indexing(images, x0, y0)
            Ib = self.advance_indexing(images, x0, y1)
            Ic = self.advance_indexing(images, x1, y0)
            Id = self.advance_indexing(images, x1, y1)

        with tf.name_scope("Interpolation"):
            x0 = tf.cast(x0, dtype=tf.float32)
            x1 = tf.cast(x1, dtype=tf.float32)
            y0 = tf.cast(y0, dtype=tf.float32)
            y1 = tf.cast(y1, dtype=tf.float32)

            wa = (x1-x) * (y1-y)
            wb = (x1-x) * (y-y0)
            wc = (x-x0) * (y1-y)
            wd = (x-x0) * (y-y0)

            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)

        return tf.math.add_n([wa*Ia + wb*Ib + wc*Ic + wd*Id])

#Local contrast normalization implementation
# taken from https://github.com/keras-team/keras/issues/2918


def lcn(x):
    ones_for_weight = np.reshape(np.ones((32, 32)), (1, 32, 32))
    mu = sum_pool2d(x, pool_size=(7, 7), strides=(1, 1), padding=(3, 3))
    mu_weight = sum_pool2d(ones_for_weight, pool_size=(
        7, 7), strides=(1, 1), padding=(3, 3))
    sum_sq_x = sum_pool2d(K.square(x), pool_size=(7, 7),
                          strides=(1, 1), padding=(3, 3))
    total_mu_sq = mu_weight * K.square(mu)
    sq_cross_term = -2 * K.square(mu)
    sigma = K.sqrt(sum_sq_x + total_mu_sq + sq_cross_term)
    return (x - mu)/(sigma + 1)


def lcn_output_shape(input_shape):
    return input_shape


def sum_pool2d(x, pool_size=(7, 7), strides=(1, 1), padding=(3, 3)):
    sum_x = pool.pool_2d(x, ds=pool_size, st=strides,
                         mode='sum', padding=padding, ignore_border=True)
    return sum_x


def sum_pool2d_output_shape(input_shape):
    return input_shape


input = keras.Input(shape=(48, 48, 3))

theta = Localization(f1=250, f2=250, f3=250)(input)
x = BilinearInterpolation(height=48, width=48)([input, theta])
x = keras.layers.ZeroPadding2D(2)(x)
x = Conv2D(200, 7, activation="relu", padding='valid')(x)
x = MaxPooling2D(2)(x)
x = BatchNormalization()(x)  # cant find LCN implementation in keras, using Batchnorm

theta = Localization(f1=200, f2=150, f3=200)(x)
x = BilinearInterpolation(height=23, width=23)([x, theta])
x = keras.layers.ZeroPadding2D(2)(x)
x = Conv2D(250, 4, activation="relu", padding='valid')(x)
x = MaxPooling2D(2)(x)
x = BatchNormalization()(x)

theta = Localization(f1=250, f2=150, f3=200)(x)
x = BilinearInterpolation(height=12, width=12)([x, theta])
x = keras.layers.ZeroPadding2D(2)(x)
x = Conv2D(350, 4, activation="relu", padding='valid')(x)
x = MaxPooling2D(2)(x)
x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(400, activation="relu")(x)
output = Dense(48, activation="softmax")(x)

model = keras.Model(input, output)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              loss='categorical_crossentropy',
              metrics='accuracy')


if training_percent == 20:
    model_no = 20
elif training_percent == 40:
    model_no = 40
elif training_percent == 60:
    model_no = 60
elif training_percent == 80:
    model_no = 80
else:
    model_no = 80
model.load_weights('saves/adam_aug_' + str(model_no) +'/adam_aug_' + str(model_no))


if(model_no == 20 or model_no == 40):

    testing_datagen = ImageDataGenerator(
        rotation_range=abs(rotating_degree),
        width_shift_range=abs(x_direction),
        height_shift_range=abs(y_direction),
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest",
        brightness_range=[0.75, 1.25],
        preprocessing_function=preprocessing_function)


    test_ds = testing_datagen.flow_from_directory('Test_final/',
                                                target_size=(48, 48), batch_size=128,
                                                class_mode='categorical')
else:
    testing_datagen = ImageDataGenerator(
        rotation_range=abs(rotating_degree),
        width_shift_range=abs(x_direction),
        height_shift_range=abs(y_direction),
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest",
        brightness_range=[0.75, 1.25],
        preprocessing_function=preprocessing_function)


    test_ds = testing_datagen.flow_from_directory('Test_final_2/',
                                                target_size=(48, 48), batch_size=128,
                                                class_mode='categorical')


met = model.metrics_names
if Augment_Image:
    results = model.evaluate(test_ds)
    print(met)
    print(results)

    # st.write(results)
    st.header(" ")
    st.header("Results")
    st.subheader("Accuracy = " + str(results[1]))
    st.subheader("Loss = " + str(results[0]))
