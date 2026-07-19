'''
Use Logistic Regression for classification
Preprocess data for modeling
Implement Logistic regression on real world data
'''
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

'''
Classification with Logistic Regression

Scenario
Assume that you are working for a telecommunications company which is concerned about the number of customers leaving their land-line business for cable competitors. 
They need to understand who is more likely to leave the company.
'''

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)

churn_df.sample(5)

'''
Let's select some features for the modeling. Also, we change the target data type to be an integer, as it is a requirement by the scikit-learn algorithm:

Data Preprocessing
For this lab, we can use a subset of the fields available to develop out model. Let us assume that the fields we use are 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip' and of course 'churn'.
'''

churn_df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip' , 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int') # converting float to int
churn_df.head(5)

'''
the target to be predicted is 'churn', the data under this field will be stored under the variable 'y'. We may use any combination or all of the remaining fields as the input. 
Store these values in the variable 'X'.
'''

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
print("first row", X[0]) # prints first row
print(X[0:5])  #print the first 5 rows
'''
first row: [ 11.  33.   7. 136.   5.   5.   0.]

[[ 11.  33.   7. 136.   5.   5.   0.]
 [ 33.  33.  12.  33.   2.   0.   0.]
 [ 23.  30.   9.  30.   1.   2.   0.]
 [ 38.  35.   5.  76.   2.  10.   1.]
 [  7.  35.  14.  80.   2.  15.   0.]]
'''
#similarly for churn

y = np.asarray(churn_df['churn'])
print(y[0:5]) #print the first 5 values
'''
[1 1 0 0 0]
'''
'''
It is also a norm to standardize or normalize the dataset in order to have all the features at the same scale.
This helps the model learn faster and improves the model performance. We may make use of StandardScalar function in the Scikit-Learn library.
'''

std_scalar = StandardScalar()
std_scalar.fit(X).transform(X)



