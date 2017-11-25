import numpy as np
import tensorflow as tf
import csv
import sys
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
csv.field_size_limit(sys.maxsize)
with open("botnetdataset3.csv", 'r') as f:
    botdata = list(list(csv.reader(f, delimiter=",",quoting=csv.QUOTE_NONE)))
i=0
X=[]
Y=[]
for i in range (len(botdata)):
    botinfo=np.array(botdata[i])
    X.append(botinfo[:10])
    Y.append(botinfo[10:11])
    #data=np.append(np.array(bot))
#X=bot[:,2]
#print(X[1:100])
#X=np.array(X[0]);
#X=int(X);
#Y=int(Y);
#Y=np.array(Y[1:9000]);
#X_train_counts = count_vect.fit_transform(X)
#X_train_counts.shape
X = np.array(X[1:])
print(X[0])
Y = np.array(Y)
print (X.shape,Y.shape)
clf=svm.SVC(decision_function_shape="ovo");
for i in range (len(Y)):
    if(Y[i]=="botnet"):
        Y[i]=1
    else:
        Y[i]=0

# print (X,Y)
clf.fit(X,Y)
a=clf.test(X[:9001,:])
print(a)
clf.test(X[:9002,:])
