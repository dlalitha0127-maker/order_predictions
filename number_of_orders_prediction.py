#imp lib
import numpy as np
import pandas as pd
#radient gradient boosting regression tree, model assembly , Beast model 
data = pd.read_csv("Supplement_Sales_Weekly_Expanded.csv")
#data.info()
#data.isnull().sum()
#data.describe() #returns descriptive statistics

import plotly.express as px


pie=data["Product Name"].value_counts().reset_index()
product=pie.index
orders=pie.values

fig=px.pie(pie, values="count", names="Product Name")
#fig.show()

pie1=data["Discount"].value_counts().reset_index()
discount=pie1.index
orders=pie1.values
fig=px.pie(pie1, values="count", names="Discount")
#fig.show()

pie2=data["Platform"].value_counts().reset_index()
platform=pie2.index
orders=pie2.values
fig=px.pie(pie2, values="count", names="Platform")
#fig.show()


data["Platform"]=data["Platform"].map({"Amazon":1, "iHerb":2, "Walmart":3})
data["Location"]=data["Location"].map({"Canada":1, "UK":2, "USA":3})

#creating variables
X=data[["Product Name", "Location", "Discount"]]
X["Product Name"] = X["Product Name"].astype("category")
X["Location"] = X["Location"].astype("category")
y=data["Units Sold"]
#print(X.head())

# building the ml model
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X ,y, test_size=0.2, random_state=42)


import lightgbm as ltb
model = ltb.LGBMRegressor()
model.fit(X_train , y_train)

# predicting results
y_pred = model.predict(X_test)
data_pred = pd.DataFrame({"Predicted orders" : y_pred.flatten()})

# printing results
results = X_test.copy()
results["Actual Units"]=y_test.values
results["Predicted Units"]=y_pred
results = results.reset_index(drop=True)
print(results)

# printing model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))








