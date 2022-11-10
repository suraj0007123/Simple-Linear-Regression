### For Reading the dataset 
### Importing the necessary Libraries 

import pandas as pd ### deals with dataFrame 
import numpy as np ### Deals with numerical values 

DT_ST = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\simple linear regressionsss\Datasets_SLR\delivery_time.csv")

DT_ST.info()

DT_ST.columns ='DeliveryTime','SortingTime'

DT_ST.info()

DT_ST.isna().sum()

duplicate=DT_ST.duplicated()
duplicate
sum(duplicate)

DT_ST.describe()

##### For Graphical Representation 

import matplotlib.pyplot as plt


plt.bar(height=DT_ST.SortingTime, x=np.arange(1,22,1)) ### Barplot 

plt.hist(DT_ST.SortingTime) ### Histogram

plt.boxplot(DT_ST.SortingTime) ### Boxplot 


plt.bar(height=DT_ST.DeliveryTime, x=np.arange(1,22,1)) ### Barplot 

plt.hist(DT_ST.DeliveryTime) ### Histogram

plt.boxplot(DT_ST.DeliveryTime) ### Boxplot 

### Scatter plot 
plt.scatter(x=DT_ST['DeliveryTime'],y=DT_ST['SortingTime'], color='green')

np.corrcoef(DT_ST.DeliveryTime,DT_ST.SortingTime) #correlation
help(np.corrcoef)

import statsmodels.formula.api as smf

model = smf.ols('SortingTime ~ DeliveryTime', data=DT_ST).fit()
model.summary()

#values prediction
#Confidence interval Calculation
pred1 = model.predict(pd.DataFrame(DT_ST['DeliveryTime']))
pred1

# Regression Line
plt.scatter(DT_ST.DeliveryTime, DT_ST.SortingTime)
plt.plot(DT_ST.DeliveryTime, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

print (model.conf_int(0.95)) # 95% confidence interval

res = DT_ST.SortingTime - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse

######### Model building on Transformed Data#############

# Log Transformation
# x = log(SortingTime); y = DeliveryTime
plt.scatter(x=np.log(DT_ST['DeliveryTime']),y=DT_ST['SortingTime'],color='brown')
np.corrcoef(np.log(DT_ST.DeliveryTime), DT_ST.SortingTime) #correlation

model2 = smf.ols('SortingTime ~ np.log(DeliveryTime)',data=DT_ST).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(DT_ST['DeliveryTime']))
pred2

# Regression Line
plt.scatter(np.log(DT_ST.DeliveryTime), DT_ST.SortingTime)
plt.plot(np.log(DT_ST.DeliveryTime), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

print(model2.conf_int(0.95)) # 95% confidence level

res2 = DT_ST.SortingTime - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2

# Exponential transformation
plt.scatter(x=DT_ST['DeliveryTime'], y=np.log(DT_ST['SortingTime']),color='orange')

np.corrcoef(DT_ST.SortingTime, np.log(DT_ST.DeliveryTime)) #correlation

model3 = smf.ols('np.log(SortingTime) ~ DeliveryTime',data=DT_ST).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(DT_ST['DeliveryTime']))
pred_log

pred3 = np.exp(pred_log)
pred3

# Regression Line
plt.scatter(DT_ST.DeliveryTime, np.log(DT_ST.SortingTime))
plt.plot(DT_ST.DeliveryTime, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

print(model3.conf_int(0.95)) # 95% confidence level

res3 = DT_ST.SortingTime - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
rmse3

############Polynomial model with 2 degree (quadratic model)  ;x = SortingTime*SortingTime; y = DeliveryTime############
#### input=x & X^2 (2-degree); output=y  SortingTime
model4 = smf.ols('SortingTime ~ DeliveryTime+I(DeliveryTime*DeliveryTime)', data=DT_ST).fit()
model4.summary()

pred_p2 = model4.predict(pd.DataFrame(DT_ST['DeliveryTime']))
pred_p2

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = DT_ST.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = DT_ST.iloc[:, 1].values


plt.scatter(DT_ST.DeliveryTime, np.log(DT_ST.SortingTime))
plt.plot(X, pred_p2, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

print(model3.conf_int(0.95)) # 95% confidence level

res4 = DT_ST.SortingTime - pred_p2
sqres4 = res4*res4
mse4 = np.mean(sqres4)
rmse4 = np.sqrt(mse4)
rmse4


###########Polynomial model with 3 degree (quadratic model)  ;x = SortingTime*SortingTime*SortingTime; y = DeliveryTime############
#### input=x & X^2 (2-degree); output=y  np.log(SortingTime)
model5 = smf.ols('SortingTime ~ DeliveryTime+I(DeliveryTime*DeliveryTime)+I(DeliveryTime*DeliveryTime*DeliveryTime)', data=DT_ST).fit()
model5.summary()

pred_p3 = model5.predict(pd.DataFrame(DT_ST['DeliveryTime']))
pred_p3

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X = DT_ST.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = DT_ST.iloc[:, 1].values


plt.scatter(DT_ST.DeliveryTime, np.log(DT_ST.SortingTime))
plt.plot(X, pred_p3, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

print(model5.conf_int(0.95)) # 95% confidence level

res5 = DT_ST.SortingTime - pred_p3
sqres5 = res5*res5
mse5 = np.mean(sqres5)
rmse5 = np.sqrt(mse5)
rmse5
