from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense,GlobalAveragePooling2D,Dropout
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import pandas as pd
import numpy as np
from itertools import chain

def mygen(batch_size,valid):
	df = pd.read_csv('train_v2.csv')
	labels = sorted(set(chain.from_iterable([tag.split() for tag in df['tags'].values])))
	y_map =  {tag: index for index, tag in enumerate(labels)}
	x_train = np.zeros([batch_size,256,256,3])
	y_train = np.zeros([batch_size,len(y_map)])
	index = 0
	if valid:
		df = df[len(df)*8//10:]
	else:
		df = df[:len(df)*8//10]
	while True:
		x_train[index%batch_size]=image.img_to_array(image.load_img('train-jpg/%s.jpg'%(df['image_name'][index%len(df)]),target_size=(256,256)))
		for tag in df['tags'][index%len(df)].split():
			y_train[index%batch_size][y_map[tag]]=1
		index +=1
		yield x_train,y_train
			
			
base_model = InceptionV3(weights='imagenet',include_top=False,input_shape=(256,256,3))
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
		generator=mygen(10,False),
		steps_per_epoch=3000,
		epochs=2,
		validation_data=mygen(10,True),
		validation_steps = 800,
		verbose=1,
		callbacks=[ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True),],
	)
