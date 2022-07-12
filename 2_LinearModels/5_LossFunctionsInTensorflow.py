"""
Loss functions in TensorFlow
In this exercise, you will compute the loss using data from the King County housing dataset. You are given a target, price, which is a tensor of house prices, and predictions, which is a tensor of predicted house prices. You will evaluate the loss function and print out the value of the loss.

Instructions 2/2
50 XP
Import the keras module from tensorflow. Then, use price and predictions to compute the mean squared error (mse).

Modify your code to compute the mean absolute error (mae), rather than the mean squared error (mse).
"""


# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean absolute error (mae)
loss = keras.losses.mae(price, predictions)

# Print the mean absolute error (mae)
print(loss.numpy())