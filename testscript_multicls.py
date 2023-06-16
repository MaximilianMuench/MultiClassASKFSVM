import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from models.askfsvm import ASKFSVM


# Define a function to generate a linear kernel
def rbf_kernel(X1, X2, gamma=1.0):
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sqdist)


def main():
    # Generate some sample data
    np.random.seed(0)  # for reproducibility
    n_samples = 100
    n_features = 3
    n_classes = 5  # change this to 2 for a binary problem

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, cluster_std=1.0, random_state=0)
    X = MinMaxScaler().fit_transform(X)

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    K0_train = rbf_kernel(X_train, X_train, 0.1)
    K0_test = rbf_kernel(X_test,X_train,   0.1)

    K1_train = rbf_kernel(X_train, X_train, 1.0)
    K1_test = rbf_kernel(X_test,X_train,   1.0)

    K2_train = rbf_kernel(X_train, X_train, 10.0)
    K2_test = rbf_kernel(X_test,X_train,   10.0)

    K3_train = rbf_kernel(X_train, X_train, 100.0)
    K3_test = rbf_kernel(X_test,X_train,   100.0)

    # Precompute the linear kernel
    K_train = [K0_train, K1_train, K2_train, K3_train]
    K_test = [K0_test, K1_test, K2_test, K3_test]

    # Initialize the SVM with maximum iterations and subsample size
    svm = ASKFSVM(max_iter=2000, subsample_size=0.6, mp=True)

    # Train the SVM
    svm.fit(K_train, y_train)

    # Use the trained SVM to make predictions on the test set
    y_pred = svm.predict(K_test)

    # Compute the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Test accuracy: {accuracy}')
    pause = "pause"


if __name__ == '__main__':
    main()
