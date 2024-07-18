import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=42)

#plot the data set
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Synthetic Dataset')
plt.show()

# train the SVM linear classifier
svm = SVC(kernel='linear')
svm.fit(xtrain, ytrain)
ypred = svm.predict(xtest)
accuracy_svm = accuracy_score(ytest,ypred)
print(f'Accuracy: {accuracy_svm*100:.2f}%')

# plot decision boundary
def pdb(clf, X, y, ax):
    xmin, xmax = X[:,0].min()-1, X[:,0].max()+1
    ymin, ymax = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(xmin,xmax, 0.01), np.arange(ymin, ymax, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:,0], X[:,1],c=y, edgecolors='k', cmap='coolwarm') 
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(clf.__class__.__name__)

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))
pdb(svm, X, y, ax1)
plt.show()