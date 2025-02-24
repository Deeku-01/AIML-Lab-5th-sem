import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target  # Features and labels
class_names = iris.target_names  # Class names (Setosa, Versicolor, Virginica)

class NaiveBayes:
    def fit(self, X, y):
        """Train the model: Compute mean, variance, and priors for each class."""
        self.classes = np.unique(y)  # Get unique class labels (0, 1, 2)
        self.mean = []  # Mean of each feature for each class
        self.var = []   # Variance of each feature for each class
        self.priors = []  # Prior probabilities of each class

        for c in self.classes:
            X_c = X[y == c]  # Select only the samples of class 'c'
            self.mean.append(X_c.mean(axis=0))  # Compute mean for each feature
            self.var.append(X_c.var(axis=0))  # Compute variance for each feature
            self.priors.append(len(X_c) / len(y))  # Compute prior probability P(class)

        # Convert lists to numpy arrays for efficient computation
        self.mean = np.array(self.mean)
        self.var = np.array(self.var)
        self.priors = np.array(self.priors)

    def predict(self, X):
        """Predict class labels for each sample in X."""
        predictions = []
        for x in X:
            predicted_class = self._classify(x)  # Predict class for each sample
            predictions.append(predicted_class)
        return np.array(predictions)

    def _classify(self, x):
        """Compute the posterior probability for each class and return the class with the highest probability."""
        posteriors = []  # Stores posterior probabilities for all classes

        for i, c in enumerate(self.classes):
            # Compute log(P(y)) -> Log prior probability of the class
            log_prior = np.log(self.priors[i])

            # Compute log(P(X|y)) -> Log likelihood using Gaussian distribution
            log_likelihood = np.sum(np.log(self._gaussian(i, x)))

            # Compute posterior probability (log scale)
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)

        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def _gaussian(self, i, x):
        """Compute the Gaussian probability density function for feature x given class i."""
        mean = self.mean[i]  # Mean of the features for class i
        var = self.var[i]  # Variance of the features for class i

        # Compute Gaussian probability density function
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator  # Returns P(X|y)

# Split the dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

# Create the Naïve Bayes classifier, train it, and make predictions
nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print class name predictions
print("Predictions:", class_names[y_pred])

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print classification report with precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))
