'''
Use scikit-learn to implement simple linear regression

Create, train, and test a linear regression model on real data

Understand the data
FuelConsumption.csv:
You will use a fuel consumption dataset, FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. Dataset source.

MODEL YEAR e.g. 2014
MAKE e.g. VOLVO
MODEL e.g. S60 AWD
VEHICLE CLASS e.g. COMPACT
ENGINE SIZE e.g. 3.0
CYLINDERS e.g 6
TRANSMISSION e.g. AS6
FUEL TYPE e.g. Z
FUEL CONSUMPTION in CITY(L/100 km) e.g. 13.2
FUEL CONSUMPTION in HWY (L/100 km) e.g. 9.5
FUEL CONSUMPTION COMBINED (L/100 km) e.g. 11.5
FUEL CONSUMPTION COMBINED MPG (MPG) e.g. 25
CO2 EMISSIONS (g/km) e.g. 182
Your task will be to create a simple linear regression model from one of these features to predict CO2 emissions of unobserved cars based on that feature.
'''
# Import needed packages
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)
df.head() # df.sample(5) - gives random 5 rows

# Explore the data
# First, consider a statistical summary of the data.
df.describe()
'''
Select features
Select a few features that might be indicative of CO2 emission to explore more.
'''
c_df = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
c_df.sample(10)

'''
Visualize features
Consider the histograms for each of these features.
'''
viz = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
viz.show()

plt.scatter(c_df.FUELCONSUMPTION_COMB , c_df.CO2EMISSIONS, color = 'blue')
plt.xlabel('fuel consumption comb')
plt.ylabel('Emissions')
plt.show()

plt.scatter(c_df.ENGINESIZE , c_df.CO2EMISSIONS, color = 'blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2_EMISSIONS")
plt.show()

'''
Practice excercise 1
Plot CYLINDERS against CO2 Emission, to see how linear their relationship is.
'''
plt.scatter(c_df.CYLINDERS, c_df.CO2EMISSIONS, color = 'blue')
plt.xlabel("CYLINDER")
plt.ylabel("CO2EMISSIONS")
plt.show()

'''
**Extract the input feature and labels from the dataset**
Although perhaps not necessarily the ideal choice of input feature, for illustration purposes, you will use engine size to predict CO2 emission with a linear regression model.
You can begin the process by extracting the input feature and target output variables, X and y, from the dataset.
'''
# converting the entire column values into an array
X = c_df.ENGINESIZE.to_numpy()
y = c_df.CO2EMISSIONS.to_numpy()
print(X)
print(y)
'''
[2.  2.4 1.5 ... 3.  3.2 3.2]
[196 221 136 ... 271 260 294]
'''

# Create train and test datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.20 , random_state = 1)

type(X_train) , np.shape(X_train) , np.shape(X_test) # (numpy.ndarray, (853,), (214,))

lr = LinearRegression()

# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
lr.fit(X_train.reshape(-1, 1), y_train)

# Print the coefficients
print ('Coefficients: ', lr.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array other it will be [38.992978724434074]
print ('Intercept: ',lr.intercept_)
'''
Coefficients:  38.992978724434074
Intercept:  126.28970217408721
Here, Coefficient and Intercept are the regression parameters determined by the model.
They define the slope and intercept of the 'best-fit' line to the training data.
'''

'''
Visualize model outputs
You can visualize the goodness-of-fit of the model to the training data by plotting the fitted line over the data.

The regression model is the line given by y = intercept + coefficient * x.
'''
plt.scatter(X_train , y_train, color = 'blue')
plt.plot(X_train , lr.coef_ * x + lr.intercept, color = 'red')
plt.xlabel('Engine size')
plt.ylabel('CO2_Emissions')
plt.show()

'''
Model evaluation
You can compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics play a key role in the development of a model, as they provide insight into areas that require improvement.

There are different model evaluation metrics, let's use MSE here to calculate the accuracy of our model based on the test set:

Mean Absolute Error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just an average error.

Mean Squared Error (MSE): MSE is the mean of the squared error. In fact, it's the metric used by the model to find the best fit line, and for that reason, it is also called the residual sum of squares.

Root Mean Squared Error (RMSE). RMSE simply transforms the MSE into the same units as the variables being compared, which can make it easier to interpret.

R2-Score is not an error but rather a popular metric used to estimate the performance of your regression model. It represents how close the data points are to the fitted regression line. The higher the R2-Score value, the better the model fits your data. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
'''

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use the predict method to make test predictions
y_pred = lr.predict(X_test.reshape(-1,1))

# Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2-score: %.2f" % r2_score(y_test, y_pred))

'''
Mean absolute error: 24.10
Mean squared error: 985.94
Root mean squared error: 31.40
R2-score: 0.76
'''

'''
Practice exercises
1. Plot the regression model result over the test data instead of the training data. Visually evaluate whether the result is good.
'''
plt.scatter(X_test, y_test, color = 'blue')
plt.plot( X_test, lr.coef_ * X_test + lr.intercept_, color = 'red')
plt.xlabel("Engine size")
plt.ylabel("C02_emmisions")
plt.show()
'''
2. Select the fuel consumption feature from the dataframe and split the data 80%/20% into training and testing sets.
Use the same random state as previously so you can make an objective comparison to the previous training result.
'''
X = c_df.FUELCONSUMPTION_COMB.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X , y , test_size= 0.20 , random_state=1)

'''
3. Train a linear regression model using the training data you created.
Remember to transform your 1D feature into a 2D array.
'''
lr = LinearRegression()
lr.fit(X_train.reshape(-1,1), y_train)

'''
4. Use the model to make test predictions on the fuel consumption testing data.¶
'''
y_pred= regr.predict(X_test.reshape(-1,1))

'''
5. Calculate and print the Mean Squared Error of the test predictions.
'''
# Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2-score: %.2f" % r2_score(y_test, y_pred))
'''
Mean absolute error: 20.50
Mean squared error: 857.80
Root mean squared error: 29.29
R2-score: 0.79

As you might expect from your exploratory analysis, the MSE is smaller when we train using FUELCONSUMPTION_COMB(857.80) rather than ENGINESIZE(985.94).

'''
