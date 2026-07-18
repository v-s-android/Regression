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

