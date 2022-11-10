##### To read the dataset importing necessary libraries 

import pandas as pd #### deals with dataframe
import numpy  as np ### deals with numerical values 

cc=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\simple linear regressionsss\Datasets_SLR\calories_consumed.csv")

cc.info()

cc.columns="weightgained","caloriesconsumed"
cc.info()

cc.describe()

cc.isna().sum()

duplicate=cc.duplicated()
duplicate
sum(duplicate)

#### Graphical Representation 

import matplotlib.pyplot as plt

plt.bar(height=cc.weightgained, x=np.arange(1,15,1))

plt.hist(cc.weightgained)

plt.boxplot(cc.weightgained)

plt.bar(height=cc.caloriesconsumed, x=np.arange(1,15,1))

plt.hist(cc.caloriesconsumed)

plt.boxplot(cc.caloriesconsumed)


plt.scatter(x=cc['weightgained'], y=cc['caloriesconsumed'], color='green')

np.corrcoef(cc.weightgained, cc.caloriesconsumed) ## correlation

help(np.corrcoef)

cov_output=np.cov(cc.weightgained, cc.caloriesconsumed)[0,1]

cov_output

import statsmodels.formula.api as smf

plt.hist(cc['weightgained'])
plt.hist(cc['caloriesconsumed'])
model=smf.ols('weightgained ~ caloriesconsumed',data=cc).fit()
model.summary()

## values prediction 
## confidence interval calculation 
pred1 = model.predict(pd.DataFrame(cc['caloriesconsumed']))

pred1

# Regression Line
plt.scatter(cc.weightgained, cc.caloriesconsumed)
plt.plot(cc.weightgained, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()


print(model.conf_int(0.95)) ### 95% confidence interval


res = cc.caloriesconsumed - pred1
sqres = res*res
mse=np.mean(sqres)
rmse = np.sqrt(mse)
rmse

######### Model building on Transformed Data#############

# Log Transformation
# x = log(Caloriesconsumed); y = Weightgained
plt.scatter(x=np.log(cc['weightgained']),y=cc['caloriesconsumed'],color='brown')
np.corrcoef(np.log(cc.weightgained), cc.caloriesconsumed) #correlation

model2 = smf.ols('caloriesconsumed ~ np.log(weightgained)',data=cc).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(cc['weightgained']))
pred2

# Regression Line
plt.scatter(np.log(cc.weightgained), cc.caloriesconsumed)
plt.plot(np.log(cc.weightgained), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()


print(model2.conf_int(0.95)) # 95% confidence level

res2 = cc.caloriesconsumed - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2

# Exponential transformation
plt.scatter(x=cc['weightgained'], y=np.log(cc['caloriesconsumed']),color='orange')

np.corrcoef(cc.weightgained, np.log(cc.caloriesconsumed)) #correlation

model3 = smf.ols('np.log(caloriesconsumed) ~ weightgained',data=cc).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(cc['weightgained']))
pred_log

pred3 = np.exp(pred_log)
pred3

# Regression Line
plt.scatter(cc.weightgained, np.log(cc.caloriesconsumed))
plt.plot(cc.weightgained, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

print(model3.conf_int(0.95)) # 95% confidence level


res3 = cc.caloriesconsumed - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
rmse3


############Polynomial model with 2 degree (quadratic model)  ;x = Caloriesconsumed*Caloriesconsumed; y = Weightgained############
#### input=x & X^2 (2-degree); output=y  ####
model4 = smf.ols('caloriesconsumed ~ weightgained+I(weightgained*weightgained)', data=cc).fit()
model4.summary()

pred_p2 = model4.predict(pd.DataFrame(cc['weightgained']))
pred_p2

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = cc.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = cc.iloc[:, 1].values


plt.scatter(cc.weightgained, np.log(cc.caloriesconsumed))
plt.plot(X, pred_p2, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

print(model3.conf_int(0.95)) # 95% confidence level

res4 = cc.caloriesconsumed - pred_p2
sqres4 = res4*res4
mse4 = np.mean(sqres4)
rmse4 = np.sqrt(mse4)
rmse4

###########Polynomial model with 3 degree (quadratic model)  ;x = Caloriesconsumed*Caloriesconsumed*Caloriesconsumed; y = Weightgained############
#### input=x & X^2 (2-degree); output=y  ####
model5 = smf.ols('weightgained ~ caloriesconsumed+I(caloriesconsumed*caloriesconsumed)+I(caloriesconsumed*caloriesconsumed*caloriesconsumed)', data=cc).fit()
model5.summary()

pred_p3 = model5.predict(pd.DataFrame(cc['caloriesconsumed']))
pred_p3


# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X = cc.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = cc.iloc[:, 1].values


plt.scatter(cc.weightgained, np.log(cc.caloriesconsumed))
plt.plot(X, pred_p3, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


print(model5.conf_int(0.95)) # 95% confidence level

res5 = cc.weightgained - pred_p3
sqres5 = res5*res5
mse5 = np.mean(sqres5)
rmse5 = np.sqrt(mse5)
rmse5
