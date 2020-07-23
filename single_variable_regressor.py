import pickle
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn import linear_model

# Input file containing data
input_file = 'data_singlevar_regr.txt'

# Read data
data = np.loadtxt(input_file, delimiter=',')
x, y = data[:, :-1], data[:, -1]

# Train and test split
num_training = int(0.8 * len(x))
num_test = len(x) - num_training

# Training data
x_train, y_train = x[:num_training], y[:num_training]

# Test data
x_test, y_test = x[num_training:], y[num_training:]

# Create linear regression object
regressor = linear_model.LinearRegression()

# Train the model using the training sets
regressor.fit(x_train, y_train)

# Predict the output
y_test_pred = regressor.predict(x_test)

# Plot outputs
plt.scatter(x_test, y_test, color='green')
plt.plot(x_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks()
plt.show()

# Compute performance metrics
print("Linear regressor performance:")
print("Mean Absolute Error =", 
    round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean Squared Error =", 
    round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median Absolute Error =", 
    round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain Variance Score =", 
    round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 Score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Model persistence
output_model_file = 'model.pkl'

# Save the output model
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# Load the model
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Perform prediction on test data
y_test_pred_new = regressor_model.predict(x_test)
print("\nNew Mean Absolute Error =", 
    round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))