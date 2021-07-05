from re import A
from keras.layers import Input,Lambda, Dense, Flatten, Conv2D,MaxPooling2D
from keras.models import Model, Sequential
from keras.applications.vgg19 import VGG19 
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE= [224, 224]
train_path= '/Dataset/Train/'
test_path= '/Dataset/Test/'

vgg19=VGG19(input_shape=IMAGE_SIZE+[3], weights='imagenet', include_top=False)
for layer in vgg19.layers:
    layer.trainable=False

folders=glob('Dataset/Train/*')
#print(folders)

x= Flatten()(vgg19.output)
prediction=Dense(len(folders),activation='softmax')(x)
model=Model(inputs=vgg19.input,outputs=prediction)
print("FINAL_MODEL")
print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy','AUC']
)

from keras.preprocessing.image import ImageDataGenerator
train_datagen= ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    
    rotation_range=0.5
)
test_datagen= ImageDataGenerator(rescale=1./255)
training_set= train_datagen.flow_from_directory(
    'Dataset/Train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test_set= test_datagen.flow_from_directory(
    'Dataset/Test',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

r= model.fit(
    training_set,
    validation_data=test_set,
    epochs=15,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

from keras.models import load_model
model.save('models/model_vgg19.h5')

y_pred= model.predict(test_set)
print(pd.DataFrame(y_pred))

y_pred= np.argmax(y_pred,axis=1)
print(pd.DataFrame(y_pred))

#To read image
img=image.load_img('\Dataset/Test/Parasite/39P4thinF_original_IMG_20150622_110115_cell_130.png',target_size=(224,224))
img2=image.load_img('Dataset/test/Uninfected/2.png',target_size=(224,224))
x= image.img_to_array(img)
x1=image.img_to_array(img2)
x/=255
x1/=255
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
x1=np.expand_dims(x1,axis=0)
img_data1=preprocess_input(x1)

yp1=model.predict(img_data)
yp2=model.predict(img_data1)

a1= np.argmax(yp1,axis=1)
a2= np.argmax(yp2,axis=1)
def finale(a):
    if(a==1):
        print("Uninfected")
    else:
        print("Infected")
finale(a1)
finale(a2)


