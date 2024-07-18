import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.inspection  import DecisionBoundaryDisplay

# Create a synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset')
plt.show()

# Create Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)

# Train the model
dt_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test,y_pred)
print(f'Decision Tree Classifier Accuracy: {accuracy*100:.2f}%')

# Visualize the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(dt_clf, filled=True, feature_names =['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'])
plt.title('Decision Tree Structure')
plt.show()

# Plot Decision Boudnary
disp = DecisionBoundaryDisplay.from_estimator(dt_clf, X, response_method='predict', alpha=0.3, cmap='coolwarm')
plt.scatter(X[:,0], X[:,1], edgecolors='k', cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary using Scikit Learn')
plt.show()