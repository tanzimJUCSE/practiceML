import pandas as p
data=p.read_excel('hiring.xlsx')

#fill the file with proper value
data.experience=data.experience.fillna("zero")

#word to number conversion
from word2number import w2n
data.experience=data.experience.apply(w2n.word_to_num)

#create the median value
import math 
mean=math.floor(data.test_score.mean())
data.test_score=data.test_score.fillna(mean)


#applying linear regression model
from sklearn.linear_model import LinearRegression
multivariate=LinearRegression()
multivariate.fit(data[['experience','test_score','interview_score' ]],data['salary($)'])
model=multivariate

#using the pickle module
import pickle 
with open('model_pickle','wb') as file:
    pickle.dump(model,file)
with open('model_pickle','rb') as file:
    saved_model=pickle.load(file)
result=saved_model.predict([[12,8,9]])  

#using the Joblib which is specially used for large data values such as numpy
from sklearn.externals import joblib
joblib.dump(model,'model_joblib')  
saved_model_joblib=joblib.load('model_joblib')
saved_model_joblib.predict([[12,8,9]])


    

