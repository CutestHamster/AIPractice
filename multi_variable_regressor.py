import numpy as np
import sklearn.metrics as sm
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# Input file containing data
input_file = 'data_multivar_regr.txt'

# Load the data from the input file
data = np.loadtxt(input_file, delimiter=',')
x, y = data[:, :-1], data[:, -1]

# Split data into training and testing
num_training = int(0.8 * len(x))
num_test = len(x) - num_training

# Training data
x_train, y_train = x[:num_training], y[:num_training]

# Test data
x_test, y_test = x[num_training:], y[num_training:]

# Create the linear regression model
linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(x_train, y_train)

# Predict the output
y_test_pred = linear_regressor.predict(x_test)

# Measure performance
print('Linear Regressor Performance:')
print('Mean Absolute Error =', 
    round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print('Mean Squared Error =', 
    round(sm.mean_squared_error(y_test, y_test_pred), 2))
print('Median Absolute Error =', 
    round(sm.median_absolute_error(y_test, y_test_pred), 2))
print('Explained Variance Score =', 
    round(sm.explained_variance_score(y_test, y_test_pred), 2))
print('R2 Score =', 
    round(sm.r2_score(y_test, y_test_pred), 2))

# Polynomial regressor
polynomial = PolynomialFeatures(degree=10)
x_train_transformed = polynomial.fit_transform(x_train)
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(x_train_transformed, y_train)
print("\nLinear Regression:\n", linear_regressor.predict(datapoint))
print("\nPolynomial Regression:\n", poly_linear_model.predict(poly_datapoint))