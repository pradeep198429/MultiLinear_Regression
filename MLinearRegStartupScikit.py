from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
import matplotlib.pyplot as plt

import pandas as pd

df =pd.read_csv("50_Startups.csv")

X= df.iloc[ :,0:4]
y= df.iloc[:,4]


le = preprocessing.LabelEncoder()
X["State"] = le.fit_transform(X["State"])


regr = linear_model.LinearRegression()
# error here if i split and pass the data  ,
X_train, X_test, y_train, y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
modeltrain=regr.fit(X_train,y_train)

# this works
regr = linear_model.LinearRegression()
model=regr.fit(X, y)
print(model.score(X,y))
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)





