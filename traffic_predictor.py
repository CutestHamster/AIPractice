import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import classification_report

# Load input data
input_file = 'traffic_data.txt'
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line[:-1].split(',')
        data.append(items)
data = np.array(data)

# Convert string data to numerical data
label_encoder = []
x_encoded = np.empty(data.shape)
for i, item in enumerate(data[0]):
    if item.isdigit():
        x_encoded[:, i] = data[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        x_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])

x = x_encoded[:, :-1].astype(int)
y = x_encoded[:, -1].astype(int)

# Split data into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, 
    random_state=5)

# Extremely Random Forests regressor
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(x_train, y_train)

# Compute the regressor performance on test data
y_pred = regressor.predict(x_test)
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))

# Testing encoding on single unknown data instance
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = [-1] * len(test_datapoint)

# Predict the output for the test datapoint
print("Predict Traffic:", int(regressor.predict([test_datapoint_encoded])[0]))