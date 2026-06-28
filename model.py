import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
df=pd.read_csv("orders.csv")
print(df.head())

# Converting date to datetime
df["Date Order was placed"] = pd.to_datetime(df["Date Order was placed"])
df["Year"] = df["Date Order was placed"].dt.year
df["Month"] = df["Date Order was placed"].dt.month
print(df[["Date Order was placed", "Year", "Month"]].head())
print(df["Month"].unique())

#X=df[["Year","Month","Quantity Ordered"]]
df["Customer Status"] = df["Customer Status"].str.upper()

df["Customer Status"] = df["Customer Status"].map({
    "SILVER": 1,
    "GOLD": 2,
    "PLATINUM": 3
})

X = df[["Quantity Ordered", "Cost Price Per Unit", "Customer Status"]]
y=df["Total Retail Price for This Order"]
model=LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
comparison = pd.DataFrame({
    "Actual Revenue": y_test,
    "Predicted Revenue": y_pred
})

print(comparison.head(10))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
print("Model trained succesfully!")

#Predict Revenue
prediction = model.predict(
    pd.DataFrame(
        [[2022, 25.2, 10]],
        columns=["Quantity Ordered", "Cost Price Per Unit", "Customer Status"])
)

print("Predicted Revenue:", prediction)

#Decision Tree Model

tree_model=DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train,y_train)
tree_pred=tree_model.predict(X_test)
tree_mae=mean_absolute_error(y_test,tree_pred)
tree_r2=r2_score(y_test,tree_pred)
print("Decision Tree MAE:" , tree_mae)
print("Decision Tree R2:" ,tree_r2)

#Random Forest Model

forest_model= RandomForestRegressor(n_estimators=100,random_state=42)
forest_model.fit(X_train,y_train)
forest_pred=forest_model.predict(X_test)
forest_mae=mean_absolute_error(y_test,forest_pred)
forest_r2=r2_score(y_test,forest_pred)
print("Random Forest MAE:" , forest_mae)
print("Random Forest R2:" ,forest_r2)
