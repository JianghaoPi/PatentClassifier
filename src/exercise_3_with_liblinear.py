import liblinearutil
import time
import matplotlib.pyplot as pyplot
import sklearn.metrics as metrics
import data
import random
def divide_problem(sizeP,sizeN):
    x_p=[]
    x_n=[]
    for i in range(len(data.x_train)):
        if data.y_train[i]==1:
            x_p.append((data.label_train[i],data.x_train[i]))
        else:
            x_n.append((data.label_train[i],data.x_train[i]))
    x_p.sort(key=lambda x:x[0])
    x_n.sort(key=lambda x:x[0])
    x_p=[i[1]for i in x_p]
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
            t.append(liblinearutil.problem([1]*len(x_p[i:i+sizeP])+[-1]*len(x_n[j:j+sizeN]), x_p[i:i+sizeP]+x_n[j:j+sizeN]))
        problems.append(t)
    return problems
problems = divide_problem(10000,10000)
parameter = liblinearutil.parameter('-s 0 -c 1')
time_start = time.clock()
models=[]
for i in range(len(problems)):
    t=[]
    for j in range(len(problems[i])):
        t.append(liblinearutil.train(problems[i][j], parameter))
    models.append(t)
print("Exercise 3 with LIBLINEAR training finished in %f."%(time.clock()-time_start))
time_start = time.clock()
p_val=[-1e100]*len(data.x_test)
for i in range(len(problems)):
    t=[1e100]*len(data.x_test)
    for j in range(len(problems[i])):
        p_label, p_acc, p_val_ij=liblinearutil.predict(data.y_test, data.x_test,models[i][j],'-b 0')
        for k in range(len(t)):
            t[k]=min(t[k],p_val_ij[k][0])
    for k in range(len(t)):
        p_val[k]=max(p_val[k],t[k])
print("Exercise 3 with LIBLINEAR predicting finished in %f."%(time.clock()-time_start))
TPR=[]
FPR=[]
for s in [-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8]:
    TP=0
    FN=0
    FP=0
    TN=0
    for i in range(len(p_label)):
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
pyplot.title("ROC of Exercise 3 with LIBLINEAR")
pyplot.savefig("roc_exercise_3_with_liblinear.pdf")
#pyplot.show()
