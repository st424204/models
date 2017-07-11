from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense,GlobalAveragePooling2D,Dropout
from keras import backend as K
import pandas as pd
import numpy as np
from itertools import chain

def mygen(batch_size):
	df = pd.read_csv('train_v2.csv')
	labels = sorted(set(chain.from_iterable([tag.split() for tag in df['tags'].values])))
	y_map =  {tag: index for index, tag in enumerate(labels)}
	x_train = np.zeros([batch_size,64,64,3])
	y_train = np.zeros([batch_size,len(y_map)])
	while True:
		index = 0
		for x in df.sample(batch_size).itertuples():
			x_train[index]=image.img_to_array(image.load_img('train-jpg/%s.jpg'%(x[1]),target_size=(64, 64)))
			for tag in x[2].split():
				y_train[index][y_map[tag]]=1
			index +=1
		yield x_train,y_train
			
			
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(64,64,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(17,activation='sigmoid')(x)
model = Model(inputs = base_model.input,output=predictions)

for layer in base_model.layers:
	layer.trainable=False
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

model.fit_generator(
		generator=mygen(128),
		steps_per_epoch=2000,
		epochs=10,
		verbose=1,
	)
model.save('model.h5')