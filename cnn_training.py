# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:51:15 2020

@author: Shikhar
"""


import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras


DATA_PATH="data.json"

def load_data(dataset_path):
    with open(dataset_path,"r") as fp:
        data=json.load(fp)
    
    x=np.array(data["mfcc"])
    y=np.array(data["labels"])
    
    return x,y

def prepare_datasets(test_size,validation_size):
    x,y=load_data(DATA_PATH)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size)
    x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=validation_size)
    x_train=x_train[...,np.newaxis]
    x_validation=x_validation[...,np.newaxis]
    x_test=x_test[...,np.newaxis]
    
    return  x_train,x_validation,x_test,y_train,y_validation,y_test

def build_model(input_shape):
    model=keras.Sequential()
    
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(32,(2,2),activation='relu',input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2),strides=(2,2),padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(10,activation='softmax'))
    
    return model

def predict(model,x,y):
    x=x[np.newaxis,...]
    prediction=model.predict(x)
    predicted_index=np.argmax(prediction,axis=1)
    print("Expected Index: {}, Predicted Index: {}".format(y, predicted_index))


if __name__=="__main__":
    
    x_train,x_validation,x_test,y_train,y_validation,y_test=prepare_datasets(0.25,0.2)
    
    input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
    model=build_model(input_shape)
    
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    
    model.fit(x_train,y_train,validation_data=(x_validation,y_validation),batch_size=32,epochs=30)
    
    test_error,test_accuracy=model.evaluate(x_test,y_test,verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    
    x=x_test[120]
    y=y_test[120]
    
    predict(model,x,y)
    
    
    
    
    