from keras.models import Sequential
from keras.layers import Dense, Activation
import random
import numpy
def train(x,y):
    def to_array(m):
        t=[0]*5000
        for i in m:
            t[i-1]=m[i]
        return t
    def generate(x,y,batch_size):
        perm=list(range(len(x)))
        random.shuffle(perm)
        i=0
        while 1:
            x_t=[to_array(x[perm[(i+j)%len(x)]])for j in range(batch_size)]
            y_t = [(y[perm[(i + j) % len(x)]]+1)/2 for j in range(batch_size)]
            yield (numpy.array(x_t),numpy.array(y_t))
            i=(i+batch_size)%len(x)
    model = Sequential()
    model.add(Dense(32, input_shape=(5000,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    batch_size=632
    model.fit_generator(generate(x,y,batch_size),
                        steps_per_epoch=len(x)//batch_size, epochs=10)
    return model
def predict(x,y,model):
    val=[]
    def to_array(m):
        t=[0]*5000
        for i in m:
            t[i-1]=m[i]
        return t
    def generate(x,y,batch_size):
        perm=range(len(x))
        random.shuffle(perm)
        i=0
        while 1:
            x_t=[to_array(x[(i+j)%len(x)])for j in range(batch_size)]
            y_t = [(y[(i + j) % len(x)]+1)/2 for j in range(batch_size)]
            yield (numpy.array(x_t),numpy.array(y_t))
            i=(i+batch_size)%len(x)
    batch_size=14
    t=model.predict_generator(generate(x,y,batch_size),
                        steps=len(x)//batch_size)
    return [2*i[0]-1for i in t]