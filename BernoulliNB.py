import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# Create synthetic dataset with binary features
X_b, y_b = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42, n_classes=2)

# Ensure features are binary
X_b = np.random.randint(0,2,size=X_b.shape)

# Split the dataset into training and testing sets
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b, test_size = 0.2, random_state=42)

# Create Bernoulli Bayes Model
bnb = BernoulliNB()

# Train the model
bnb.fit(X_train_b, y_train_b)

# Make the prediction
y_pred_bnb = bnb.predict(X_test_b)

# Evaluate model
accuracy_bnb = accuracy_score(y_test_b, y_pred_bnb)
print(f'Accuracy: {accuracy_bnb*100:.2f}%')

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
plot_decision_boundary(bnb, X_b, y_b, ax)
plt.show()
