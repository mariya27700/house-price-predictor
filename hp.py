import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df=pd.read_csv("houses.csv")
print("Average price: ",df["price"].mean())
print("Highest price:",df["price"].max())
print("Lowest price:",df["price"].min())
#expensive house
hp=df["price"].max()
exp_house=df[df["price"]==hp]
print(exp_house)
lp=df["price"].min()
low_house=df[df["price"]==lp]
print(low_house)

# luxury
for i in range(len(df)):
    if df["price"][i]>=5000000:
        print(df["price"][i],"(price) is a luxury house with rooms: ",df["rooms"][i])
    else:
        print(df["price"][i],"(price) is not a luxury house with rooms: ",df["rooms"][i])

#mean,variance,std
print("Mean:",np.mean(df["price"]))
print("Variance:",np.var(df["price"]))
print("Standard deviation:",np.std(df["price"]))

#bar
a=df["size"]
b=df["price"]
plt.bar(a,b)
plt.title("Size vs price")
plt.xlabel("size")
plt.ylabel("price")
plt.show()

#scatter
a1=df["rooms"]
b=df["price"]
plt.scatter(a1,b,color="pink")
plt.title("Room vs price")
plt.xlabel("room")
plt.ylabel("price")
plt.show()

#hist
plt.hist(b,color="violet",bins=5)
plt.title("Price distribution")
plt.show()
#pie
ep=0
ip=0
for i in df["price"]:
    if i>=5000000:
        ep+=1
    else:
        ip+=1
s=[ep,ip]
plt.pie(s,labels=["expensive","affordable"],colors=["lightgreen","pink"])
plt.title("Expensive vs affordable")
plt.legend()
plt.show()

#boxplot
plt.boxplot(b)
plt.title("Price Spread")
plt.show()

#ml-linear regression
X=df[["size","rooms","age"]]
y=df["price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=LinearRegression()
model.fit(X_train,y_train)
new_house=pd.DataFrame({
    "size":[2000],
    "rooms":[3],
    "age":[5]
})
p=model.predict(new_house)

#accuracy
score=model.score(X_test,y_test)
print(score)
print("New price:",p)
y_pred = model.predict(X_test)
print("MAE:",mean_absolute_error(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("R2 score:",r2_score(y_test,y_pred))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE:",rmse)
