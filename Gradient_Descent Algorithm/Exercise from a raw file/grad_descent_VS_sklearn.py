import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
data_file=pd.read_csv("test_scores.csv")
data_math=np.array(data_file[['math']])
data_cs=np.array(data_file[['cs']])


def linear_regression(x,y):
    lreg=LinearRegression()
    lreg.fit(x,y)
    return lreg.coef_ , lreg.intercept_

def grad_descent(x,y):
    m_curr=c_curr=0
    n=len(x)
    step_rate=0.01
    itr=100000
    cost_pre=0
    for i in range(itr):
        y_predicted=m_curr*x + c_curr
        cost=(1/n)*sum([value**2 for value in (y-y_predicted)])
        slope_der=-(2/n)*sum(x*(y-y_predicted))
        intercept_der=-(2/n)*sum(y-y_predicted)
        m_curr=m_curr-(step_rate*slope_der)
        c_curr=m_curr-(step_rate*intercept_der)
        if (math.isclose(cost,cost_pre,rel_tol=1e-15)):
            break
        cost_pre=cost
    
    return m_curr , c_curr,i

grad_slope,grad_inter,iteration=grad_descent(data_math,data_cs) 
linear_slope,linear_inter=linear_regression(data_math,data_cs)
print(f"GRADIENT_SLOPE{grad_slope}  GRADIENT_INT{grad_inter} ITERATION{iteration}")
print(f"LINEAR_SLOPE{linear_slope}  LINEAR_INT{linear_inter}")       
    
