from multiprocessing import Pool
import liblinearutil
import time
import matplotlib.pyplot as pyplot
import sklearn.metrics as metrics
import data
import random
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
parallel_computing=0
def divide_problem(sizeP,sizeN,data_only=0):
    x_p = []
    x_n = []
    for i in range(len(data.x_train)):
        if data.y_train[i] == 1:
            x_p.append((data.label_train[i], data.x_train[i]))
        else:
            x_n.append((data.label_train[i], data.x_train[i]))
    x_p.sort(key=lambda x: x[0])
    x_n.sort(key=lambda x: x[0])
    x_p = [i[1] for i in x_p]
    x_n = [i[1] for i in x_n]
    if sizeP>len(x_p) or sizeN>len(x_n):
        print("Warning: sizeP or sizeN too large!")
        sizeP=min(len(x_p),sizeP)
        sizeN = min(len(x_n), sizeN)
    problems=[]
    for i in range(0,len(x_p),sizeP):
        t=[]
        for j in range(0,len(x_n),sizeN):
            print("Problem %d,%d: %d,%d"%(i,j,len(x_p[i:i+sizeP]),len(x_n[j:j+sizeN])))
            if data_only==0:
                t.append(liblinearutil.problem([1]*len(x_p[i:i+sizeP])+[-1]*len(x_n[j:j+sizeN]), x_p[i:i+sizeP]+x_n[j:j+sizeN]))
            else:
                t.append(([1]*len(x_p[i:i+sizeP])+[-1]*len(x_n[j:j+sizeN]), x_p[i:i+sizeP]+x_n[j:j+sizeN]))
        problems.append(t)
    return problems
def parallel_train_predict(args):
    print("A process begins.")
    x_train,y_train,x_test,y_test=args
    problem = liblinearutil.problem(y_train, x_train)
    parameter = liblinearutil.parameter('-s 0 -c 1')
    time_start = time.clock()
    model = liblinearutil.train(problem, parameter)
    print("A process training finished in %f."%(time.clock()-time_start))
    time_start = time.clock()
    p_label, p_acc, p_val = liblinearutil.predict(y_test, x_test,model,'-b 0')
    print("A process predicting finished in %f."%(time.clock()-time_start))
    return p_val
if parallel_computing==0:
    problems = divide_problem(20000,20000,data_only=1)
    parameter = liblinearutil.parameter('-s 0 -c 1')
    models=[]
    time_start = time.clock()
    for i in range(len(problems)):
        t=[]
        for j in range(len(problems[i])):
            t.append(train(problems[i][j][1],problems[i][j][0]))
        models.append(t)
    print("Exercise 3 with MLP training finished in %f."%(time.clock()-time_start))
    time_start = time.clock()
    p_val=[-1e100]*len(data.x_test)
    for i in range(len(problems)):
        t=[1e100]*len(data.x_test)
        for j in range(len(problems[i])):
            p_val_ij=predict(data.x_test, data.y_test,models[i][j])
            for k in range(len(t)):
                t[k]=min(t[k],p_val_ij[k])
        for k in range(len(t)):
            p_val[k]=max(p_val[k],t[k])
    print("Exercise 3 with MLP predicting finished in %f."%(time.clock()-time_start))
else:
    time_start = time.clock()
    problems = divide_problem(20000,20000,data_only=1)
    flat_ploblems=[]
    models=[]
    for i in range(len(problems)):
        for j in range(len(problems[i])):
            flat_ploblems+=[(problems[i][j][1],problems[i][j][0],data.x_test,data.y_test)]
    p = Pool(4)
    flat_predicts=p.map(parallel_train_predict,flat_ploblems)
    p_val=[-1e100]*len(data.x_test)
    uu=0
    for i in range(len(problems)):
        t=[1e100]*len(data.x_test)
        for j in range(len(problems[i])):
            p_val_ij=flat_predicts[uu]
            uu+=1
            for k in range(len(t)):
                t[k]=min(t[k],p_val_ij[k][0])
        for k in range(len(t)):
            p_val[k]=max(p_val[k],t[k])
    print("Exercise 3 with MLP training and predicting finished in %f."%(time.clock()-time_start))
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
        print("Accuracy: %f, Precision: %f, Recall: %f, F1: %f" % ((TP + TN) / (TP + TN + FP + FN), p, r, 2 * r * p / (r + p)))
print("AUC: %f" % metrics.auc(FPR, TPR))
pyplot.plot(FPR, TPR, lw=1)
pyplot.grid(True)
pyplot.xlim([0, 1])
pyplot.ylim([0, 1])
pyplot.xlabel("FPR")
pyplot.ylabel("TPR")
pyplot.title("ROC of Exercise 3 with MLP")
pyplot.savefig("roc_exercise_3_with_mlp.pdf")
#pyplot.show()
