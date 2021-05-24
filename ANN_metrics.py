from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot


X=array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
model=Sequential()
model.add(Dense(2,input_dim=1))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['mse','mae','mape','cosine_proximity','accuracy'])
history=model.fit(X,X,epochs=100,batch_size=len(X),verbose=2)
pyplot.plot(history.history['mse'])
pyplot.plot(history.history['mae'])
pyplot.plot(history.history['mape'])
pyplot.plot(history.history['cosine_proximity'])
pyplot.plot(history.history['accuracy'])
pyplot.show()
