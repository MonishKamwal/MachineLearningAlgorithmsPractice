print("Perceptron")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X,y= make_classification(n_samples = 100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=42)

# plot the data set
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset')
plt.show()

#train the perceptron
perceptron= Perceptron(max_iter=1000, eta0=0.01, random_state=42)

perceptron.fit(xtrain,ytrain)
ypred= perceptron.predict(xtest)
accuracy = accuracy_score(ytest, ypred)

print(f'Accuracy : {accuracy*100:.2f}%')

#visulaize the decision boundary

fig, ax  = plt.subplots()
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z= perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
ax.scatter(X[:,0], X[:,1], c=y, edgecolors='k', cmap='coolwarm')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Perceptron Decision Boundary')
plt.show()