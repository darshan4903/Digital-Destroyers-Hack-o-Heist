# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:51:08 2021

@author: Hp
"""
#importing
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# giving names to two classes
X = "cats"
Y = "dogs"

# picking up the sample image to work on = directory name
sample_X_image = "train/X/cat.13.jpg"

#Create a function that will tweak our imags to prevent overfitting
# it will generate bunch of images of the existing image
datagen = ImageDataGenerator(
            rotation_range=40,  #upto 40 degrees
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            rescale = 1.0/255,
            shear_range=0.2,  #shift left or right
            zoom_range=0.2,
            horizontal_flip=True, #allow image to flip
            fill_mode='nearest')

# load that image
img = load_img(sample_X_image)


x = img_to_array(img)
x = x.reshape((1,) + x.shape) # make sure that the image is square otherwise it will get distorted

# creating no. of images
i = 0

for batch in datagen.flow(x,
                          batch_size=1,
                          save_to_dir='preview',
                          save_prefix=Y,
                          save_format='jpeg'):
    i += 1
    if i > 20:
        break