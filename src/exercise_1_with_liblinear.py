import liblinearutil
import time
import matplotlib.pyplot as pyplot
import sklearn.metrics as metrics
import data
problem = liblinearutil.problem(data.y_train, data.x_train)
parameter = liblinearutil.parameter('-s 0 -c 1')
time_start = time.clock()
model = liblinearutil.train(problem, parameter)
print("Exercise 1 with LIBLINEAR training finished in %f."%(time.clock()-time_start))
time_start = time.clock()
p_label, p_acc, p_val = liblinearutil.predict(data.y_test, data.x_test,model,'-b 0')
print("Exercise 1 with LIBLINEAR predicting finished in %f."%(time.clock()-time_start))
TPR=[]
FPR=[]
for s in [-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8]:
    TP=0
    FN=0
    FP=0
    TN=0
    for i in range(len(p_label)):
        y_p=1 if p_val[i][0]>s else -1
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
        print("Precision: %f, Recall: %f, F1: %f" % (p,r,2*r*p/(r+p)))
print("AUC: %f" % metrics.auc(FPR, TPR))
pyplot.plot(FPR, TPR, lw=1)
pyplot.grid(True)
pyplot.xlim([0, 1])
pyplot.ylim([0, 1])
pyplot.xlabel("FPR")
pyplot.ylabel("TPR")
pyplot.title("ROC of Exercise 1 with LIBLINEAR")
pyplot.savefig("roc_exercise_1_with_liblinear.pdf")
pyplot.show()
