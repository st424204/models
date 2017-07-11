from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense,GlobalAveragePooling2D,Dropout
from keras.optimizers import SGD
from keras import backend as K
import pandas as pd
import numpy as np
from itertools import chain

def mygen(batch_size):
	df = pd.read_csv('train_labels.csv')
	x_train = np.zeros([batch_size,64,64,3])
	y_train = np.zeros([batch_size,1])
	while True:
		index = 0
		for x in df.sample(batch_size).itertuples():
			x_train[index]=image.img_to_array(image.load_img('train/%s.jpg'%(x[1]),target_size=(64, 64)))
			y_train[index][0]=x[2]
			index +=1
		yield x_train,y_train
			
	
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(64,64,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1,activation='sigmoid')(x)
model = Model(inputs = base_model.input,output=predictions)

for layer in base_model.layers:
	layer.trainable=False
	
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

model.fit_generator(
		generator=mygen(128),
		steps_per_epoch=20,
		epochs=50,
		verbose=1,
	)
model.save('fc.h5')



model = load_model('fc.h5')
for layer in model.layers[:25]:
	layer.trainable=False
for layer in model.layers[25:]:
	layer.trainable=True
	
model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),loss='binary_crossentropy',metrics=['acc'])
model.fit_generator(
		generator=mygen(128),
		steps_per_epoch=20,
		epochs=50,
		verbose=1,
	)
model.save('model.h5')