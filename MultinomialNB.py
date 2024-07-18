
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import accuracy_score


# Create data with integer features
X_m, y_m = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
# emsure features are positive integers
X_m= np.abs(X_m)
X_m = np.ceil(X_m).astype(int)

# split data into training  and testing sets
X_train_m , X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size=0.2, random_state=42)

# Create the multinomial Naive Bayes model
mnb = MultinomialNB()

# train model
mnb.fit(X_train_m, y_train_m)

# make prediction 
y_pred_m = mnb.predict(X_test_m)

# Evaluate model
accuracy_mnb = accuracy_score(y_test_m, y_pred_m)
print(f'Multinomial Naive Bayes Accuracy: {accuracy_mnb*100:.2f}%')

# Plot the decision boundary
def plot_decision_boundary(clf, X, y, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(clf.__class__.__name__)

# Plot the decision boundaries
fig, ax = plt.subplots(figsize=(6, 5))
plot_decision_boundary(mnb, X_m, y_m, ax)
plt.show()
