import pandas as pd


df=pd.read_csv("Computer_Data.csv")
print(df.head())
X= df.iloc[:,2:11]
y= df.iloc[:,1]


X2= pd.get_dummies(X,columns=['cd','multi','premium'])
print(X2.head())

import statsmodels.api as sm
import matplotlib.pyplot as plt

X = sm.add_constant(X2)
print(X.head)
model= sm.OLS(y,X).fit()
print(model.summary())
sm.graphics.influence_plot(model)
plt.show();

X3 = sm.add_constant(X2)
X=X3.drop(X3.index[[5960,1440,1700]],axis=0)
y=y.drop(y.index[[5960,1440,1700]],axis=0)
model2= sm.OLS(y,X).fit()
sm.graphics.influence_plot(model2)
plt.show();
print(model2.summary())