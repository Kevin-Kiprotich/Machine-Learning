import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pickle
data=pd.read_csv('biomassdata.csv')
print(data.head(10))
x=data[['LAIL81','EVI','NDVI','DVI','SAVI']]
y=data['bio2']
# normally the dataset has to be scaled
rgr=RandomForestRegressor(n_estimators=5000)
rgr.fit(x,y)
# you can save the regression model or used it directly in your code
# To use it directly, use the following code
predicted=rgr.predict(x)
#to save it externally use the followig code
pickle.dump(rgr, open('RF_biomodel.sav', 'wb'))
loaded=pickle.load(open('RF_biomodel.sav','rb'))
predicted=loaded.predict(x)
print(predicted)
print(loaded.score(x,y))



