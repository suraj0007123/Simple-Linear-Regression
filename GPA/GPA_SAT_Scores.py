### For reading the dataset
### Importing the libraries 

import pandas as pd 
import numpy as np

data=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\simple linear regressionsss\Datasets_SLR\SAT_GPA.csv")

data.info()

data.describe()

data.isna().sum()

duplicate=data.duplicated()
duplicate
sum(duplicate)

data1=data.drop_duplicates()

### For Graphical Representation
import matplotlib.pyplot as plt

plt.bar(height=data.GPA,x=np.arange(1,201,1))

plt.hist(data.GPA)

plt.boxplot(data.GPA)

plt.bar(height=data.SAT_Scores, x=np.arange(1,201,1))

plt.hist(data.SAT_Scores)

plt.boxplot(data.SAT_Scores)

#### Scatterplot

plt.scatter(x=data['SAT_Scores'], y=data['GPA'], color='green')

### Correlation 
np.corrcoef(data.SAT_Scores, data.GPA)

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('GPA ~ SAT_Scores', data = data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(data['SAT_Scores']))

# Regression Line
plt.scatter(data.SAT_Scores, data.GPA)
plt.plot(data.SAT_Scores, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = data.GPA - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(data['SAT_Scores']), y = data['GPA'], color = 'brown')
np.corrcoef(np.log(data.SAT_Scores), data.GPA) #correlation

model2 = smf.ols('GPA ~ np.log(SAT_Scores)', data = data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(data['SAT_Scores']))
pred2

# Regression Line
plt.scatter(np.log(data.SAT_Scores), data.GPA)
plt.plot(np.log(data.SAT_Scores), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = data.GPA - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = data['SAT_Scores'], y = np.log(data['GPA']), color = 'orange')
np.corrcoef(data.SAT_Scores, np.log(data.GPA)) #correlation

model3 = smf.ols('np.log(GPA) ~ SAT_Scores', data = data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(data['SAT_Scores']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(data.SAT_Scores, np.log(data.GPA))
plt.plot(data.SAT_Scores, pred3_at, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = data.GPA - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(GPA) ~ SAT_Scores + I(SAT_Scores*SAT_Scores)', data = data).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(data))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = data.iloc[:, 1].values


plt.scatter(data.SAT_Scores, np.log(data.GPA))
plt.plot(X, pred4_at, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = data.GPA - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4
