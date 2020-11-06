# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:34:57 2020

@author: pasproj-admin
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:46:17 2020

@author: pasproj-admin
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:15:33 2020

@author: pasproj-admin
"""



import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


data = pd.read_csv("C:\\data\\50_Startups.csv")
print(data.shape)


inputvariables=list(data)
del inputvariables[4]
inputvariables

datacor =data.corr()
outputvariables=list(data)[4]
outputvariables
inputData=data[inputvariables]

#regr = linear_model.LinearRegression()
#regr.fit(X, Y)

catcolumns=['State']



for column in catcolumns:
    dummyCols=pd.get_dummies(inputData[column])#for each column in catcolumns its creating dummy values
    inputData=inputData.join(dummyCols)#after that it joining the dummy columns to the input dataset
    del inputData[column]#after that original column for which dummy variable is created will be deleted here
#print(inputData)
outputData=data[[outputvariables]]

import statsmodels.api as sm
import statsmodels.formula.api as smf

inputData = sm.add_constant(inputData)
model=sm.OLS(outputData,inputData).fit()
model.summary()
sm.graphics.influence_plot(model)
plt.show();


inputData_new1=inputData.drop(inputData.index[[45,48,46,49]],axis=0)

outputData_new1=outputData.drop(outputData.index[[45,48,46,49]],axis=0)

inputData_new1 = sm.add_constant(inputData_new1)
model_new1=sm.OLS(outputData_new1,inputData_new1).fit()
model_new1.summary()
sm.graphics.influence_plot(model_new1)
plt.show();
print("hi")

inputData_new2=inputData_new1.drop(inputData_new1.index[[15,14,36]],axis=0)

outputData_new2=outputData_new1.drop(outputData_new1.index[[15,14,36]],axis=0)

inputData_new2 = sm.add_constant(inputData_new2)
model_new2=sm.OLS(outputData_new2,inputData_new2).fit()
model_new2.summary()
sm.graphics.influence_plot(model_new2)


inputData_new3=inputData_new2.drop(inputData_new2.index[[12]],axis=0)
outputData_new3=outputData_new2.drop(outputData_new2.index[[12]],axis=0)

inputData_new3 = sm.add_constant(inputData_new3)
model_new3=sm.OLS(outputData_new3,inputData_new3).fit()
model_new3.summary()

sm.graphics.influence_plot(model_new3)
#so our final model is model_new3