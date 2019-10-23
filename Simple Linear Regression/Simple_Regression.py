#libraries are imported 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data file is read
dataset=pd.read_excel('data.xlsx')

#data is splitted as independent and dependent variables
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,1].values

# dividing the complete dataset into training and test dataaset
from sklearn.model_selection._split import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#implement our classifier based on simple linear regression
from sklearn.linear_model import LinearRegression
simplelinearRegression=LinearRegression()
simplelinearRegression.fit(X_train,Y_train)
Y_predict=simplelinearRegression.predict(X_test)
#Y_predict_ll=simplelinearRegression.predict([[11]])


#implement the graph
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,simplelinearRegression.predict(X_train))
plt.show()