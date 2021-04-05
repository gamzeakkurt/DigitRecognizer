import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

#read train file
train=pd.read_csv("train.csv")

#read test file
test=pd.read_csv("test.csv")

#read submission file
sample_submission = pd.read_csv('sample_submission.csv')

#get train label column 
labels=train['label']

#drop label column from train file
train = train.drop(['label'], axis=1)

#convert image to 2 dimensional array
train=train.to_numpy().reshape(-1,28,28)
test = test.to_numpy().reshape(-1,28,28)

#the size of figure
plt.figure(figsize=(10,10))

#fill the figure the size of 5*5
for i in range(25):
    #first row, second column , third ith graphic
    plt.subplot(5,5,i+1)
    #get first image the size of 28*28
    plt.imshow(train[i])
    plt.xticks([])
    plt.yticks([])
    #get the labels of digit
    plt.xlabel(labels[i])

#save the figure
plt.savefig("Numbers.png")

#show the plot
plt.show()

#create bar plots for digits
fig, ax = plt.subplots(figsize=(8,8))
sns.countplot(labels)
ax.set_title('Distribution of Digits', fontsize=14)
ax.set_xlabel('Digits', fontsize=12)
ax.set_ylabel('Total Number of Digits', fontsize=14)

#save the figure
plt.savefig("NumberDistribution.png")

#show the figure
plt.show()


#expand of shape of array horizontal to vertical e.g: 1,2 -> 1
#                                                            2
train = np.expand_dims(train,axis=-1)
labels = np.expand_dims(labels,axis=-1)
test = np.expand_dims(test,axis=-1)


#padding feature maps to increase 32x32 dimensions
train = tf.pad(train,[[0,0],[2,2],[2,2],[0,0]])
test = tf.pad(test,[[0,0],[2,2],[2,2],[0,0]])

#split train and test data
num = train.shape[0]//10
train_data, val_data, test_data = tf.split(train,[num*8, num, num])
train_label, val_label, test_label = tf.split(labels,[num*8, num, num])


# Create neural network model. 
image_shape = (32,32,1)

inputs = layers.Input(shape=image_shape)

#Normalized feature maps
x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
#Use convolution neural network filter 32
x = layers.Conv2D(32,kernel_size=(3,3),activation='relu')(x) 
x = layers.MaxPooling2D(pool_size=(2,2))(x) 
x = layers.Conv2D(48,kernel_size=(3,3),activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(256,kernel_size=(3,3),activation='relu')(x)
x = layers.Flatten()(x)
#fully connected layer
x = layers.Dense(84,activation='relu')(x)  
x = layers.Dropout(0.5)(x)
#output layer
outputs = layers.Dense(10)(x) 
model = keras.Model(inputs,outputs)
#model summary
model.summary()
#Compiling model using adam optimizer, loss function and accuracy metric
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

#Stop training when accuracy is not improved 20 consecutive times
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='auto',
                                    patience=20,restore_best_weights=True)

#reducing learning rate when a metric has stopped improving 5 consecutive times
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',mode='auto',factor=0.5,patience=5)

#fitting model
history = model.fit(train_data,train_label,batch_size=32,epochs=30,validation_data=(val_data,val_label),
                    callbacks=[early_stopping,lr_scheduler])

#evaluate model on test data
model.evaluate(test_data,test_label,verbose=2)

#predict test label 
result =  model.predict(test)

#returns indices of max element for predicted label
#predict_label = np.argmax(result,axis=-1)

#submission file
#sample_submission['Label'] = predict_label
#sample_submission.to_csv('submission.csv', index=False)
