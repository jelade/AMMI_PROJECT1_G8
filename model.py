def add_ones(w):

  one = np.ones((len(w))).reshape(len(w),1)

  w = np.hstack((one,w))
  return w

## Add ones to X
X= add_ones(X)
X

def split_data(X,Y, train_size):
  # ADD YOUR CODES
  # shuffle the data before splitting it
  x_train_size = round(len(X)* train_size)
  x_test_size = len(X) - x_train_size
  
  np.random.shuffle(X)
  np.random.shuffle(Y)
  X_train = X[:x_train_size]
  X_test = X[x_train_size:]
  y_train = y[:x_train_size]
  y_test = y[x_train_size:]


  return X_train, X_test, y_train, y_test

# Split (X,y) into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test= split_data(X,y,0.8)

def mse(y, y_pred):
  error = y- y_pred
  ms = np.dot(error,error)/len(y)
  return ms

def normalEquation(X,y):
    # ADD YOUR CODES

    theta_hat = np.linalg.inv(np.transpose(X) @ X) @ (np.transpose(X)@y)
    return theta_hat

b = normalEquation(X_train,y_train)
print(b)

def predict(x,a):
  y = x@a
  return y

Y = predict(X_test,b)

u = mse(y_test,Y)
u

class LinearRegression:

  def __init__(self, model= "lin"):
      # ADD YOUR CODES
     self.model = model
  def fit(self,x,y):
      # ADD YOUR CODES
      self.x = x
      self.y = y
      if self.model == "lin":
        self.theta = normalEquation(self.x,self.y)
      elif self.model == "cgs":
        self.theta = cgss(self.x,self.y)
      else:
        return "Unknown estimator"
        
        #norma
    
  def predict(self,x):
      #ADD YOUR CODES
      y_predict = x @ self.theta

      return y_predict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Instanciate the LinearRegression class 
model= LinearRegression("cgs")

model1 = LinearRegression()

# Train the model

model.fit(X_train, y_train)
model1.fit(X_train, y_train)

# print the learned theta

print(model.theta)
print(model1.theta)

# Make a prediction on X_test

y_pp = model.predict(X_test)
y_pp1 = model1.predict(X_test)

print(y_pp)


print(y_pp1)

# Compute the MSE (Evaluate both, regression and classification)

normal = mse(y_test,y_pp)
print(normal)

normal1 = mse(y_test,y_pp1)

print(normal1)

