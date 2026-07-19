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

std_scalar = StandardScaler()
#X_norm =  std_scalar.fit(X).transform(X)
X_std = std_scalar.fit_transform(X)
X_std[0:5]

'''
array([[-1.13518441, -0.62595491, -0.4588971 ,  0.4751423 ,  1.6961288 , -0.58477841, -0.85972695],
       [-0.11604313, -0.62595491,  0.03454064, -0.32886061, -0.6433592 , -1.14437497, -0.85972695],
       [-0.57928917, -0.85594447, -0.261522  , -0.35227817, -1.42318853, -0.92053635, -0.85972695],
       [ 0.11557989, -0.47262854, -0.65627219,  0.00679109, -0.6433592 , -0.02518185,  1.16316   ],
       [-1.32048283, -0.47262854,  0.23191574,  0.03801451, -0.6433592 , 0.53441472, -0.85972695]])
 '''

'''
Splitting the dataset
The trained model has to be tested and evaluated on data which has not been used during training. 
Therefore, it is required to separate a part of the data for testing and the remaining for training. For this, we may make use of the train_test_split
function in the scikit-learn library.
'''

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.20 , random_state = 1 )

# Logistic Regression Classifier modeling
'''
Let's build the model using LogisticRegression from the Scikit-learn package and fit our model with train data set.
'''
LR = LogisticRegression()
LR.fit(X_train , y_train)

'''
Fitting, or in simple terms training, gives us a model that has now learnt from the traning data and can be used to predict the output variable.
Let us predict the churn parameter for the test data set.
'''
y_pred = LR.predict(X_test)
print(y_pred[:10]) # array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
'''
To understand this prediction, we can also have a look at the prediction probability of data point of the test data set. Use the function predict_proba ,
we can get the probability of each class. The first column is the probability of the record belonging to class 0, and second column that of class 1.
Note that the class prediction system uses the threshold for class prediction as 0.5. This means that the class predicted is the one which is most likely.
'''

y_proba = LR.predict_proba(X_test)
print(y_proba[:10]) 

'''
array([[0.74643946, 0.25356054],
       [0.92667894, 0.07332106],
       [0.83442627, 0.16557373],
       [0.94600618, 0.05399382],
       [0.84325532, 0.15674468],
       [0.71448367, 0.28551633],
       [0.77076426, 0.22923574],
       [0.90955642, 0.09044358],
       [0.26152115, 0.73847885],
       [0.94900731, 0.05099269]])
'''

'''
Since the purpose here is to predict the 1 class more acccurately, you can also examine what role each input feature has to play in the prediction of the 1 class.
Consider the code below.
'''
LR.coef_ # array([[-0.84569763, -0.17566042, -0.12422107, -0.01015039,  0.06012391,  -0.23290551,  0.75198953]])

coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

'''
Performance Evaluation
Once the predictions have been generated, it becomes prudent to evaluate the performance of the model in predicting the target variable.
Let us evaluate the log-loss value.

 log loss
 Log loss (Logarithmic loss), also known as Binary Cross entropy loss, is a function that generates a loss value based on the class
 wise prediction probabilities and the actual class labels. The lower the log loss value, the better the model is considered to be.
'''

print("the log loss is :",log_loss(y_test, yhat_prob)) # 0.6257718410257235

'''
a. Let us assume we add the feature 'callcard' to the original set of input features. What will the value of log loss be in this case?

Hint
Reuse all the code statements above after modifying the value of churn_df. Make sure to edit the list of features feeding the variable X. The expected answer is 0.6039104035600186.
b. Let us assume we add the feature 'wireless' to the original set of input features. What will the value of log loss be in this case?

Hint
Reuse all the code statements above after modifying the value of churn_df. Make sure to edit the list of features feeding the variable X. The expected answer is 0.7227054293985518.
c. What happens to the log loss value if we add both "callcard" and "wireless" to the input features?

Hint
Reuse all the code statements above after modifying the value of churn_df. Make sure to edit the list of features feeding the variable X. The expected answer is 0.7760557225417114
d. What happens to the log loss if we remove the feature 'equip' from the original set of input features?

Hint
Reuse all the code statements above after modifying the value of churn_df Make sure to edit the list of features feeding the variable X. The expected answer is 0.5302427350245369
e. What happens to the log loss if we remove the features 'income' and 'employ' from the original set of input features?

Hint
Reuse all the code statements above after modifying the value of churn_df. Make sure to edit the list of features feeding the variable X. The expected answer is 0.6529317169884828.
'''
