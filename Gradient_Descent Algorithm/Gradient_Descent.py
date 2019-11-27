import numpy as np
def grad_descent(x,y):
    n=len(x)
    iterations=1000
    m_curr=c_curr=0
    step_rate=0.02
    for itr in range(iterations):
        y_predicted=m_curr*x + c_curr
        cost=(1/n)*sum([value**2 for value in (y-y_predicted)])
        intercept_der=-(2/n)*sum(y-y_predicted)
        slope_der=-(2/n)*sum(x*(y-y_predicted))
        m_curr=m_curr-(step_rate*slope_der)
        c_curr=m_curr-(step_rate*intercept_der)
        print(f"COST{cost}  INTERCEPT{c_curr} SLOPE{m_curr}")
        
x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
grad_descent(x,y)

