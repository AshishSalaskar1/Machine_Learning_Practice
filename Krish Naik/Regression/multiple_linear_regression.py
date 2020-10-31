import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
data = pd.read_csv("50_startups.csv")
x = data.iloc[:,:-1]
y = data.iloc[:,-1]


# One hot Encoding
state = pd.get_dummies(x['State'],drop_first=True)
x = x.drop('State',axis=1)
x = pd.concat([x,state],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

from sklearn.metrics import accuracy_score,r2_score
score = regressor.score(x_test,y_test)
r2 = r2_score(y_test, y_pred)
