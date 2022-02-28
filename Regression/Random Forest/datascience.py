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

rgr=RandomForestRegressor(n_estimators=5000)
rgr.fit(x,y)

pickle.dump(rgr, open('RF_biomodel.sav', 'wb'))

loaded=pickle.load(open('RF_biomodel.sav','rb'))
predicted=loaded.predict(x)
print(predicted)
print(loaded.score(x,y))

model=MLPRegressor(hidden_layer_sizes=7,solver='adam',max_iter=100000,activation='logistic')
model.fit(x,y)   
predic=model.predict(x)
model.score(x,y)


