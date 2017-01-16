## author - NM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Dataset.csv', header = 0)
data = data.values
x = data[:,0]
y = data[:,1]
x = np.reshape(x, (x.shape[0] , 1))
y = np.reshape(y, (y.shape[0] , 1))

#train model on data
reg = linear_model.LinearRegression(fit_intercept = True, normalize = True)
reg.fit(x, y)
predicted_values = (reg.predict(x))
print ((min(abs(predicted_values - y)))**2)

#visualize results
# plt.scatter(data[:,0] , data[:,1])
# plt.plot(x, predicted_values)
# plt.show()

