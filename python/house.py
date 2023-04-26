#read data
import numpy as np
import pandas as pd
data = pd.read_csv("C:\\xampp\\htdocs\\HOUSE_PRICE_PREDICTION\\data.csv")

x = data.drop(labels = ["price"],axis=1).copy()
y = ((data["price"])/1000000).copy()

#preprocssing using minmaxscaler
from sklearn.preprocessing import MinMaxScaler
algo = MinMaxScaler()
algo.fit(x)
data1 = pd.DataFrame(data = algo.fit_transform(x),columns=x.columns,index = x.index)

#split data
from sklearn import model_selection
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size =0.2)

#train model -- XGBClassifier from xgboost
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(xtrain,ytrain)

#checking error for training data
from sklearn import metrics
pred = model.predict(xtrain)
score1 = metrics.r2_score(ytrain,pred)
score2 = metrics.mean_absolute_error(ytrain,pred)

#checking error for testing data
test_pred = model.predict(xtest)
score3 = metrics.r2_score(ytest,test_pred)
score4 = metrics.mean_squared_error(ytest,test_pred)


# load and dump the model using pickle
import pickle
pickle.dump(model,open('python/models/model.pkl','wb'))

models = pickle.load(open('python/models/model.pkl','rb'))
d = pd.DataFrame([[3,2.5,1460,7573,2,0,0,3,1460,0,1983,2009]],columns=x.columns)

print(model.predict(d)*1000000)
print(score3,score4)