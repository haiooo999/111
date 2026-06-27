import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

train = pd.read_csv(r"C:\Users\海鸥\Desktop\car_competition\used_car_train_20200313.csv", sep=" ")
test = pd.read_csv(r"C:\Users\海鸥\Desktop\car_competition\used_car_testA_20200313.csv", sep=" ")
submit = pd.DataFrame({"SaleID": test["SaleID"]})

train = train.replace("-", np.nan)
test = test.replace("-", np.nan)

train["kilometer"] = train["kilometer"].fillna(train["kilometer"].median())
test["kilometer"] = test["kilometer"].fillna(test["kilometer"].median())

le = LabelEncoder()
cat_cols = ["brand", "gearbox"]
for col in cat_cols:
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

train["car_age"] = 2026 - train["regDate"]
test["car_age"] = 2026 - test["regDate"]

drop_cols = ["price", "SaleID", "regDate", "name"]
X = train.drop(drop_cols, axis=1)
y = train["price"]
X_test = test.drop(["SaleID", "regDate", "name"], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=80, random_state=42)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, val_pred)
print(f"验证集MAE（平均绝对误差）：{mae:.2f}")

test_pred = model.predict(X_test)
submit["price"] = test_pred
submit.to_csv(r"C:\Users\海鸥\Desktop\car_competition\submit.csv", index=False)
print("提交文件 submit.csv 已生成在car_competition文件夹，可上传天池竞赛！")