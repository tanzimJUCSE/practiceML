#Problem statement : Multivariate simple linear regression model is applied by 
#where you have to show the prediction for two given inputs that's why there is no need for 
#split the dataset 
#This following hiring excel file contains many empty values 

#read the data file
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
first_result=multivariate.predict([[2,9,6]])