import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


# Extracting the data from train dataset
train_data = pd.read_csv('train.csv')
train_data = train_data.dropna(subset=['SalePrice'])
train_data = pd.get_dummies(train_data)

X_train_1 = train_data.drop('SalePrice', axis=1)
y_train_1 = train_data['SalePrice']
X_train_1 = X_train_1.fillna(0)

model = LinearRegression()
model.fit(X_train_1, y_train_1)

# Extracting the data from test dataset
test_data = pd.read_csv('test.csv')
test_data = pd.get_dummies(test_data)
X_test_1 = test_data.reindex(columns=X_train_1.columns, fill_value=0)
X_test_1 = X_test_1.fillna(0)
test_ids = test_data['Id']

# Predicting and displaying (without index) the Saleprice for each house ID in the test dataset
predictions = model.predict(X_test_1)
SalePrice = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
print(SalePrice.to_string(index=False))

# Using the model to validate the train data subset
X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_train_1, y_train_1, test_size=0.2, random_state=42)
model.fit(X_train_2, y_train_2)
y_val_predict = model.predict(X_val_2)

# Calculating rmse of actual outcome with predicted outcome
rmse = np.sqrt(mean_squared_error(y_val_2, y_val_predict))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
