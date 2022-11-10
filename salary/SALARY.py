#### For reading data set
### importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

Sy_Exp = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\simple linear regressionsss\Datasets_SLR\Salary_Data.csv")

Sy_Exp.info()

Sy_Exp.describe()

Sy_Exp.isna().sum()

duplicate=Sy_Exp.duplicated()
duplicate
sum(duplicate)

Sy_Exp.info()

#### For Graphical Representation
import matplotlib.pylab as plt #for different types of plots

plt.bar(height=Sy_Exp.Salary, x=np.arange(1,31,1))
plt.hist(Sy_Exp.Salary)
plt.boxplot(Sy_Exp.Salary)

plt.bar(height=Sy_Exp.YearsExperience, x=np.arange(1,31,1))
plt.hist(Sy_Exp.YearsExperience)
plt.boxplot(Sy_Exp.YearsExperience)

#### Scatter PLot
plt.scatter(x=Sy_Exp['YearsExperience'], y=Sy_Exp['Salary'],color='green')# Scatter plot

np.corrcoef(Sy_Exp.YearsExperience, Sy_Exp.Salary) #correlation

help(np.corrcoef)

import statsmodels.formula.api as smf

model = smf.ols('Salary ~ YearsExperience', data=Sy_Exp).fit()
model.summary()

#values prediction
#Confidence interval Calculation
pred1 = model.predict(pd.DataFrame(Sy_Exp['YearsExperience']))
pred1

# Regression Line
plt.scatter(Sy_Exp.YearsExperience, Sy_Exp.Salary)
plt.plot(Sy_Exp.YearsExperience, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()


print (model.conf_int(0.95)) # 95% confidence interval

res = Sy_Exp.Salary - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse

######### Model building on Transformed Data#############

# Log Transformation
# x = log(YearsExperience); y = Salary
plt.scatter(x=np.log(Sy_Exp['YearsExperience']),y=Sy_Exp['Salary'],color='brown')
np.corrcoef(np.log(Sy_Exp.YearsExperience), Sy_Exp.Salary) #correlation

model2 = smf.ols('Salary ~ np.log(YearsExperience)',data=Sy_Exp).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(Sy_Exp['YearsExperience']))
pred2

# Regression Line
plt.scatter(np.log(Sy_Exp.YearsExperience), Sy_Exp.Salary)
plt.plot(np.log(Sy_Exp.YearsExperience), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

print(model2.conf_int(0.95)) # 95% confidence level

res2 = Sy_Exp.Salary - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2

# Exponential transformation
plt.scatter(x=Sy_Exp['YearsExperience'], y=np.log(Sy_Exp['Salary']),color='orange')

np.corrcoef(Sy_Exp.YearsExperience, np.log(Sy_Exp.Salary)) #correlation

model3 = smf.ols('np.log(Salary) ~ YearsExperience',data=Sy_Exp).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(Sy_Exp['YearsExperience']))
pred_log

pred3 = np.exp(pred_log)
pred3

# Regression Line
plt.scatter(Sy_Exp.YearsExperience, np.log(Sy_Exp.Salary))
plt.plot(Sy_Exp.YearsExperience, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

print(model3.conf_int(0.95)) # 95% confidence level

res3 = Sy_Exp.Salary - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
rmse3

############Polynomial model with 2 degree (quadratic model)  ;x = YearsExperience*YearsExperience; y = Salary############
#### input=x & X^2 (2-degree); output=y  ####
model4 = smf.ols('Salary ~ YearsExperience+I(YearsExperience*YearsExperience)', data=Sy_Exp).fit()
model4.summary()

pred_p2 = model4.predict(pd.DataFrame(Sy_Exp['YearsExperience']))
pred_p2

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = Sy_Exp.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = Sy_Exp.iloc[:, 1].values


plt.scatter(Sy_Exp.YearsExperience, np.log(Sy_Exp.Salary))
plt.plot(X, pred_p2, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

print(model3.conf_int(0.95)) # 95% confidence level

res4 = Sy_Exp.Salary - pred_p2
sqres4 = res4*res4
mse4 = np.mean(sqres4)
rmse4 = np.sqrt(mse4)
rmse4

###########Polynomial model with 3 degree (quadratic model)  ;x = YearsExperience*YearsExperience*YearsExperience; y = Salary############
#### input=x & X^2 (2-degree); output=y  ####
model5 = smf.ols('Salary ~ YearsExperience+I(YearsExperience*YearsExperience)+I(YearsExperience*YearsExperience*YearsExperience)', data=Sy_Exp).fit()
model5.summary()

pred_p3 = model5.predict(pd.DataFrame(Sy_Exp['YearsExperience']))
pred_p3

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X = Sy_Exp.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = Sy_Exp.iloc[:, 1].values


plt.scatter(Sy_Exp.YearsExperience, np.log(Sy_Exp.Salary))
plt.plot(X, pred_p3, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


print(model5.conf_int(0.95)) # 95% confidence level

res5 = Sy_Exp.Salary - pred_p3
sqres5 = res5*res5
mse5 = np.mean(sqres5)
rmse5 = np.sqrt(mse5)
rmse5 



