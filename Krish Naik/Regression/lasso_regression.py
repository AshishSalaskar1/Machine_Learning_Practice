#!/usr/bin/env python

from sklearn.datasets import load_boston



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = load_boston()


data = pd.DataFrame(df.data)
data.columns = df.feature_names
data["price"] = df.target



x = data.iloc[:,:-1]
y = data.iloc[:,-1]


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_regressor = LinearRegression()
# lin_regressor.fit(x,y)
# cv = cross validation
mse = cross_val_score(lin_regressor, x , y, scoring='neg_mean_squared_error',cv=5)
mean_mse = np.mean(mse)
print(mean_mse)



from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

### Ridge Regression
ridge = Ridge()
# 1e-15 : 10^8 - 15
params = {"alpha" : [1e-15,1e-10,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor = GridSearchCV(ridge,params,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x,y)


# best value of alpha
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


#### Lasso Regression


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[26]:


lasso = Lasso()
# 1e-15 : 10^8 - 15
params = {"alpha" : [1e-15,1e-10,1e-2,1,5,15,50,55,60,80]}
lasso_regressor = GridSearchCV(lasso,params,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(x,y)


# In[27]:


# best value of alpha
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[29]:


pred_lasso = lasso_regressor.predict(x_test)
pred_ridge = ridge_regressor.predict(x_test)


# ### Distance Plots

# In[30]:


import seaborn as sns
sns.distplot(y_test - pred_lasso)
import distplot as plt


# In[31]:


sns.distplot(y_test - pred_ridge)


# In[ ]:





# In[ ]:





