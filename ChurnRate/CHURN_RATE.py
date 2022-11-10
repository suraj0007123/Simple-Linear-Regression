# For reading data set
# importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

SH_CO = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\simple linear regressionsss\Datasets_SLR\emp_data.csv")

SH_CO.info()

SH_CO.describe()

SH_CO.isna().sum()

duplicate = SH_CO.duplicated()
duplicate
sum(duplicate)

#### For Graphical Representation 
import matplotlib.pylab as plt #for different types of plots

SH_CO.info()

plt.bar(height=SH_CO.Churn_out_rate, x=np.arange(1,11,1))
plt.hist(SH_CO.Churn_out_rate)
plt.boxplot(SH_CO.Churn_out_rate)

plt.bar(height=SH_CO.Salary_hike, x=np.arange(1,11,1))
plt.hist(SH_CO.Salary_hike)
plt.boxplot(SH_CO.Salary_hike)

SH_CO.Salary_hike

#### ScatterPlot
plt.scatter(x=SH_CO['Salary_hike'], y=SH_CO['Churn_out_rate'],color='green')# Scatter plot

np.corrcoef(SH_CO.Salary_hike, SH_CO.Churn_out_rate) #correlation

help(np.corrcoef)

import statsmodels.formula.api as smf

plt.hist(SH_CO["Salary_hike"])

plt.hist(SH_CO["Churn_out_rate"])

model = smf.ols('Churn_out_rate ~ Salary_hike', data=SH_CO).fit()
model.summary()

#values prediction
#Confidence interval Calculation
pred1 = model.predict(pd.DataFrame(SH_CO['Salary_hike']))
pred1

print (model.conf_int(0.95)) # 95% confidence interval

# Regression Line
plt.scatter(SH_CO.Salary_hike,SH_CO.Churn_out_rate)
plt.plot(SH_CO.Salary_hike ,pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
#### From above diagram we can see the regression line between x=Salary_hike , y=Churn_out_rate(target variable) is perfect Negative Relationship


res = SH_CO.Churn_out_rate - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse

######### Model building on Transformed Data#############

# Log Transformation
# x = log(Salary_hike); y = Churn_out_rate
plt.scatter(x=np.log(SH_CO['Salary_hike']),y=SH_CO['Churn_out_rate'],color='brown')
np.corrcoef(np.log(SH_CO.Salary_hike), SH_CO.Churn_out_rate) #correlation

model2 = smf.ols('Churn_out_rate ~ np.log(Salary_hike)',data=SH_CO).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(SH_CO['Salary_hike']))
pred2
print(model2.conf_int(0.95)) # 95% confidence level


# Regression Line
plt.scatter(np.log(SH_CO.Salary_hike), SH_CO.Churn_out_rate)
plt.plot(np.log(SH_CO.Salary_hike), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
#### From above diagram and from another model logrithmic model we can see the regression line between x=Salary_hike , y=Churn_out_rate(target variable) is still perfect Negative Relationship

res2 = SH_CO.Churn_out_rate - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2

############Polynomial model with 2 degree (quadratic model)  ;x = Salary_hike*Salary_hike; y = Churn_out_rate############
#### input=x & X^2 (2-degree); output=y  ####
model4 = smf.ols('Churn_out_rate ~ Salary_hike+I(Salary_hike*Salary_hike)', data=SH_CO).fit()
model4.summary()

pred_p2 = model4.predict(pd.DataFrame(SH_CO['Salary_hike']))
pred_p2


# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = SH_CO.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = SH_CO.iloc[:, 1].values

plt.scatter(SH_CO.Salary_hike, np.log(SH_CO.Churn_out_rate))
plt.plot(X, pred_p2, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

print(model4.conf_int(0.95)) # 95% confidence level

res3 = SH_CO.Churn_out_rate - pred_p2
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
rmse3

###########Polynomial model with 3 degree (quadratic model)  ;x = Salary_hike*Salary_hike*Salary_hike; y = Churn_out_rate############
#### input=x & X^2 (2-degree); output=y  ####
model5 = smf.ols('Churn_out_rate ~ Salary_hike+I(Salary_hike*Salary_hike)+I(Salary_hike*Salary_hike*Salary_hike)', data=SH_CO).fit()
model5.summary()

pred_p3 = model5.predict(pd.DataFrame(SH_CO['Salary_hike']))
pred_p3

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X = SH_CO.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = SH_CO.iloc[:, 1].values

plt.scatter(SH_CO.Salary_hike, np.log(SH_CO.Churn_out_rate))
plt.plot(X, pred_p3, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


print(model5.conf_int(0.95)) # 95% confidence level

res4 = SH_CO.Churn_out_rate - pred_p3
sqres4 = res4*res4
mse4 = np.mean(sqres4)
rmse4 = np.sqrt(mse4)
rmse4 

