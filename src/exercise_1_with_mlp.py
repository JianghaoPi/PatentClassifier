import liblinearutil
import time
import matplotlib.pyplot as pyplot
import sklearn.metrics as metrics
import data
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
    model.add(Dense(256, input_shape=(5000,)))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    batch_size=316
    model.fit_generator(generate(x,y,batch_size),
                        steps_per_epoch=len(x)//batch_size, epochs=1)
    return model
def predict(x,y,model):
    val=[]
    def to_array(m):
        t=[0]*5000
        for i in m:
            t[i-1]=m[i]
        return t
    def generate(x,y,batch_size):
        i=0
        while 1:
            x_t=[to_array(x[(i+j)%len(x)])for j in range(batch_size)]
            y_t = [(y[(i + j) % len(x)]+1)/2 for j in range(batch_size)]
            yield (numpy.array(x_t),numpy.array(y_t))
            i=(i+batch_size)%len(x)
    batch_size=14
    t=model.predict_generator(generate(x,y,batch_size),
                        steps=len(x)//batch_size,)
    if len(t)!=len(y):
        print('error!')
        exit(0)
    return [2*i[0]-1for i in t]
time_start = time.clock()
model = train(data.x_train,data.y_train)
print("Exercise 1 with mlp training finished in %f."%(time.clock()-time_start))
time_start = time.clock()
p_val = predict(data.x_test, data.y_test,model)
print("Exercise 1 with mlp predicting finished in %f."%(time.clock()-time_start))
TPR=[]
FPR=[]
for s in [-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8]:
    TP=0
    FN=0
    FP=0
    TN=0
    for i in range(len(p_val)):
        y_p=1 if p_val[i]>s else -1
        if y_p==1:
            if y_p==data.y_test[i]:
                TP+=1
            else:
                FP+=1
        else:
            if y_p==data.y_test[i]:
                TN+=1
            else:
                FN+=1
    print('s %f TP %f FN %f FP %f TN %f'%(s,TP,FN,FP,TN))
    TPR+=[TP/(TP+FN)]
    FPR+=[FP/(FP+TN)]
    if s==0:
        p=TP/(TP+FP)
        r=TP/(TP+FN)
        print("Accuracy: %f, Precision: %f, Recall: %f, F1: %f" % ((TP+TN)/(TP+TN+FP+FN),p,r,2*r*p/(r+p)))
print("AUC: %f" % metrics.auc(FPR, TPR))
pyplot.plot(FPR, TPR, lw=1)
pyplot.grid(True)
pyplot.xlim([0, 1])
pyplot.ylim([0, 1])
pyplot.xlabel("FPR")
pyplot.ylabel("TPR")
pyplot.title("ROC of Exercise 1 with MLP")
pyplot.savefig("roc_exercise_1_with_mlp.pdf")
#pyplot.show()
