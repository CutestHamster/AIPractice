import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

# Load housing data
data = datasets.load_boston()

# Shuffle the data
x, y = shuffle(data.data, data.target, random_state=7)

# Split the data into training and testing datasets
num_training = int(0.8 * len(x))
x_train, y_train = x[:num_training], y[:num_training]
x_test, y_test = x[num_training:], y[num_training:]

# Create Support Vector Regression model
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)

# Train SVR
sv_regressor.fit(x_train, y_train)

# Evaluate performance of SVR
y_test_pred = sv_regressor.predict(x_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print('\n#### Performance ####')
print('Mean Squared Error =', round(mse, 2))
print('Explained Variance Score =', round(evs, 2))

# Test the regressor on a test datapoint
test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 
    351.34, 15.27]
print('\nPredicted Price:', sv_regressor.predict([test_data])[0])