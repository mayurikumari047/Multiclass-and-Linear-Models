import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from numpy import *
import datasets
from sklearn.neural_network import MLPClassifier

if not datasets.Quizbowl.loaded:
    datasets.loadQuizbowl()

print('\n\nRUNNING ON SMALL DATA\n')
    
print('training ava')
X = datasets.QuizbowlSmall.X
Y = datasets.QuizbowlSmall.Y
ava = OneVsRestClassifier(MLPClassifier(alpha=1)).fit(X, Y)
print('predicting ava')
avaDevPred = ava.predict(datasets.QuizbowlSmall.Xde)
print('error = {0}'.format(mean(avaDevPred != datasets.QuizbowlSmall.Yde)))

savetxt('predictionsQuizbowlSmall.txt', avaDevPred)
