import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df=pd.read_csv("housing.csv")
# view first 5 rows
print(df.head())

# check dataset information
print(df.info())
# check missing values
print(df.isnull().sum())

# check duplicated values
print(df.duplicated().sum())
# statistical summary
print(df.describe())

# convert
df=pd.get_dummies(df,drop_first=True)
# define
x = df.drop('price',axis=1)
y = df["price"]
# split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# create model
model=LinearRegression()
# train model
model.fit(x_train,y_train)
# prediction
y_pred=model.predict(x_test)
print(y_pred,2)

# act vs pred
for act,pred in zip(y_test,y_pred):
    print('Actual:',act,'Predicted:',round(pred,2))
# evaluation
mae=mean_absolute_error(y_test,y_pred)
print('mae:',mae)
mse=mean_squared_error(y_test,y_pred)
print('mse:',mse)
rmse=np.sqrt(mse)
print('rmse:',rmse)
r2=r2_score(y_test,y_pred)
print('r2:',r2)

# visual
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred)
plt.plot([y.min(),y.max()],[y.min(),y.max()])
plt.xlabel('Actual')
plt.ylabel('Predict')
plt.title('Actual vs Prediction')
plt.show()

# add new house 
new_house=pd.DataFrame({"area":[80000],"bedrooms":[4],"bathrooms":[4],"stories":[3],"parking":[3],
 "mainroad_yes":[1],"guestroom_yes":[1],"basement_yes":[0],"hotwaterheating_yes":[1],
 "airconditioning_yes":[1],"prefarea_yes":[1], "furnishingstatus_semi-furnished":[1],
 'furnishingstatus_unfurnished':[0]})  

# predicted
price = model.predict(new_house)
print("predicted new house price:",price[0])

# coef
coef_df=pd.DataFrame({"feature":x.columns,"coef":model.coef_})
print(coef_df.sort_values(by='coef',ascending=False))
# save data
import pickle
with open("house_price_model.pkl","wb") as f:
    pickle.dump(model,f)

