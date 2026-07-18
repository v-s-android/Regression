'''
Use scikit-learn to implement multiple linear regression

Create, train, and test a multiple linear regression model on real data

Understand the data
FuelConsumption.csv:
You will download and use a fuel consumption dataset, FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. Dataset source

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

create a multiple linear regression model using some of these features to predict CO2 emissions of unobserved cars based on the selected features.
'''
# !pip install numpy==2.2.0
# !pip install pandas==2.2.3
# !pip install scikit-learn==1.6.0
# !pip install matplotlib==3.9.3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)

df.head()

'''
Explore and select features
Let's select a few features to work with that might be predictive of CO2 emissions.
'''
df.describe()

# Droping non-numernical columns for now, but n practice, you would analyze these features if required to improve the accuracy of your model
df  = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE'], axis = 1) # list of columns and axis = 1 as it is a column operation
df.sample(5)
'''
Now that you have eliminated some features, take a look at the relationships among the remaining features.

Analyzing a correlation matrix that displays the pairwise correlations between all features indicates the level of independence between them.

It also indicates how predictive each feature is of the target.

You want to eliminate any strong dependencies or correlations between features by selecting the best one from each correlated group.
'''
df.corr()

'''
CO2EMISSIONS	0.874154	0.849685	0.898039	0.861748	0.892129	-0.906394	1.000000
'''

df = df.drop( ['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB'] , axis = 1)
df.sample(5)
'''
To help with selecting predictive features that are not redundant, consider the following scatter matrix, which shows the scatter plots for each pair of input features.
The diagonal of the matrix shows each feature's histogram.
'''
axes = pd.plotting.scatter_matrix(df, alpha=0.2)
# need to rotate axis labels so we can read them
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()


'''
Extract the input features and labels from the data set
Extract the required columns and convert the resulting dataframes to NumPy arrays.
'''
X = df.iloc[:,[0,1]].to_numpy() # conveting the EngineSize, FUELCONSUMPTION_COMB_MPg into numpy array
y = df.iloc[:,[2]].to_numpy() # converting the CO2Emmisions to numpy array

'''
Preprocess selected features
You should standardize your input features so the model doesn't inadvertently favor any feature due to its magnitude. 
The typical way to do this is to subtract the mean and divide by the standard deviation. Scikit-learn can do this for you.
'''
from sklearn.preprocessing import StandardScalar

std_scaler = StandardScalar()
X_std = std_scaler.fit_transform(X)
'''
In practice, if you want to properly evaluate your model, you should definitely not apply such operations to the entire dataset but to the train and test data separately.
There's more to it than that. You'll dive deeper into this and other advanced evaluation pitfalls later in the course.
'''
pd.DataFrame(X_std).describe().round(2) # we are coverting the numpy array to dataframe and then dedcribe()
'''
	        0      	1
count	1067.00	1067.00
mean	0.00	-0.00
std	1.00	1.00
min	-1.66	-2.07
25%	-0.95	-0.73
50%	0.04	-0.06
75%	0.67	0.61
max	3.57	4.50
As you can see, a standardized variable has zero mean and a standard deviation of one.
'''

'''
Create train and test datasets
 - Randomly split your data into train and test sets, using 80% of the dataset for training and reserving the remaining 20% for testing.
Build a multiple linear regression model
 - Multiple and simple linear regression models can be implemented with exactly the same scikit-learn tools.
'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.20, random_state = 1)

lr = LinearRegression()
lr.fit( X_train  , y_train) 

coef_ = lr.coef_
intercept_ = lr.intercept_

# the coefficiant and intercept
print("the Coefficients: ", lr.coef_) # there are two coeffients as we are taking EngineSize, FUELCONSUMPTION_COMB_MPg
print("Intercept:", lr.intercept_)

'''
Coefficients:  [[ 25.27339614 -37.4381472 ]]
Intercept:  [256.29072488]

The Coefficients and Intercept parameters define the best-fit hyperplane to the data. Since there are only two variables, hence two parameters, the hyperplane is a plane.
But this best-fit plane will look different in the original, unstandardized feature space.

You can transform your model's parameters back to the original space prior to standardization as follows. This gives you a proper sense of what they mean in terms of your original input features.
Without these adjustments, the model's outputs would be tied to an abstract,
transformed space that doesn’t align with the actual independent variables and the real-world problem you’re solving.
'''
# Get the standard scaler's mean and standard deviation parameters
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

# The least squares parameters can be calculated relative to the original, unstandardized feature space as:
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

print ('Coefficients: ', coef_original)
print ('Intercept: ', intercept_original)
'''
Coefficients:  [[18.51930924 -4.88859447]]
Intercept:  [323.76587705]
'''

plt.scatter(X_train[:,0], y_train,  color='blue')
plt.plot(X_train[:,0], coef_[0,0] * X_train[:,0] + intercept_[0], '-r') # coef_[0,0] from [[18.51930924 -4.88859447]] is 18.51930924
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(X_train[:,1], y_train,  color='blue')
plt.plot(X_train[:,1], coef_[0,1] * X_train[:,1] + intercept_[0], '-r') # coef_[0,1] from  [[18.51930924 -4.88859447]] is -4.88859447
plt.xlabel("FUELCONSUMPTION_COMB_MPG")
plt.ylabel("Emission")
plt.show()

'''
Exercise 1
Determine and print the parameters for the best-fit linear regression line for CO2 emission with respect to engine size.
'''
from sklearn.linear_model import LinearRegression

X_train_1 = X_train[:,0]

lr = LinearRegression()
lr.fit(X_train_1.reshape(-1,1), y_train)

coef_1 =  lr.coef_
intercept_1 = lr.intercept_

print ('Coefficients: ',coef_1)
print ('Intercept: ',intercept_1)
'''
Coefficients:  [[55.78187587]]
Intercept:  [256.66618567]
'''

'''
Exercise 2
Produce a scatterplot of CO2 emission against ENGINESIZE and include the best-fit regression line to the training data.
'''
X_train_1 = X_train[:,0] # all rows and first column ie, Engine size of training data set

plt.scatter(X_train_1 , y_train , color ="blue")
plt.plot(X_train_1 , coef_1[0] * X_train_1 + intercept_1 , color = 'red') # y = mx + c
plt.xlabel("EngineSize")
plt.ylabel("CO2Emissions")
plt.show()

'''
Exercise 3
Generate the same scatterplot and best-fit regression line, but now base the result on the test data set. Consider how the test result compares to the training result.
'''
X_test_1 = X_test[:,0] # all rows and first column ie, Engine size of test data set

plt.scatter(X_test_1, y_test,  color='blue')
plt.plot(X_test_1, coef_1[0] * X_test_1 + intercept_1, '-r')
plt.xlabel("Test: Engine size")
plt.ylabel("Test : CO2 Emission test")
plt.show()

'''
Exercise 4
Repeat the same modeling but use FUELCONSUMPTION_COMB_MPG as the independent variable instead. Display the model coefficients including the intercept.
'''
X_train_2 = X_train[:,1] # selecting the FUELCONSUMPTION_COMB_MPG

lr = LinearRegression()
lr.fit(X_train_2.reshape(-1,1), y_train)

coef_fuel = lr.coef_
intercept_fuel = lr.intercept_

print("the coefficient of  FUELCONSUMPTION_COMB_MPG ", coef_fuel )
print("the intercept of  FUELCONSUMPTION_COMB_MPG ", intercept_fuel )
'''
the coefficient of  FUELCONSUMPTION_COMB_MPG  [[-57.43890997]]
the intercept of  FUELCONSUMPTION_COMB_MPG  [256.29512547]
'''

'''
Exercise 5
Generate a scatter plot showing the results as before on the test data. Consider well the model fits, and what you might be able to do to improve it.
'''

X_test_2 = X_test[:,1] # selecting the FUELCONSUMPTION_COMB_MPG from X_test dataset
plt.scatter(X_test_2, y_test ,  color='blue')
plt.plot(X_test_2, coef_fuel[0] * X_test_2 + intercept_fuel , '-r')
plt.xlabel("Test: FUELCONSUMPTION_COMB_MPG")
plt.ylabel("Test: CO2 Emission")
plt.show()
