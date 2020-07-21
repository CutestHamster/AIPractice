import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3], [-1.2, 7.8, -6.1], 
    [3.9, 0.4, 2.1], [7.3, -9.9, -4.5]])

#Binarize and print result
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nBinarized Data:\n", *data_binarized)

#Mean Removal
#Print mean and std deviation
print("\nBefore:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

#Remove mean and print results
data_scaled = preprocessing.scale(input_data)
print("\nAfter:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

#Min max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

#L1 and L2 Normalization
data_normalized_l1 = preprocessing.normalize(input_data, norm="l1")
data_normalized_l2 = preprocessing.normalize(input_data, norm="l2")
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)