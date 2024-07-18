import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=42)

#plot the data set
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Synthetic Dataset')
plt.show()

# train the  GaussianNB model
gnb = GaussianNB()
gnb.fit(xtrain, ytrain)
ypred = gnb.predict(xtest)
accuracy_svm = accuracy_score(ytest,ypred)
print(f'Accuracy: {accuracy_svm*100:.2f}%')

# Plot the decision boundary
disp = DecisionBoundaryDisplay.from_estimator(gnb, X, response_method='predict', alpha=0.3, cmap='coolwarm')
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolors='k' )
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary using Scikit-learn')
plt.show()
