#!/usr/bin/env python
# coding: utf-8

# In[1016]:


from pathlib import Path
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[1017]:


df_train = pd.read_parquet(Path("data") / "train.parquet")


# In[1018]:


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X['week'] = X["date"].dt.isocalendar().week
    
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


# In[1019]:


def _merge_external_data(X):
    file_path = Path('data') / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    
    X['date'] = pd.to_datetime(X['date']).astype('datetime64[ns]')
    df_ext['date'] = pd.to_datetime(df_ext['date']).astype('datetime64[ns]')
    
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t", 'pmer', 'rafper', 'u', 'hnuage2', 'tn12', 'cod_tend', 'rr24', 'rr12']].sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


# In[1020]:


df_train = _merge_external_data(df_train)


# In[1021]:


X_dates_encoding = _encode_dates(df_train[["date"]])
df_train = pd.concat([df_train, X_dates_encoding], axis=1)


# In[1022]:


from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
encoder.fit(df_train[["counter_name"]])
df_train["counter_name"] = encoder.transform(df_train[["counter_name"]])


# In[1023]:


y = df_train['log_bike_count']
X = df_train.drop('log_bike_count', axis=1)


# from sklearn.preprocessing import OneHotEncoder
# 
# encoder = OneHotEncoder(sparse=False)
# data_encoded = encoder.fit_transform(X['counter_name'])
# X=

# In[1024]:


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, random_state=2408
)


# In[1025]:


selected_features = ['latitude', 't', 'u', 'month', 'weekday', 'hour', 'day', "counter_name"]
X_train = X_train[selected_features]

X_valid = X_valid[selected_features]


# In[1026]:


from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from xgboost import XGBRegressor

# Assuming df is your DataFrame containing the required columns
# Replace 'df' with the actual variable containing your dataset


# Initialize and train a Gradient Boosting Regressor
# learning_rate=0.01, max_depth=7, n_estimators=300
pipe =  XGBRegressor(learning_rate=0.22, max_depth=6, n_estimators=600, gamma=0.01, min_child_weight=7, reg_alpha=0.15)
pipe.fit(X_train, y_train)


# In[1027]:


print(
    f"Train set, RMSE={mean_squared_error(y_train, pipe.predict(X_train), squared=False):.2f}"
)
print(
    f"Train set, RMSE={mean_squared_error(y_valid, pipe.predict(X_valid), squared=False):.2f}"
)


# In[1034]:


df_test = pd.read_parquet(Path("data") / "final_test.parquet")


# In[1035]:


df_test = _merge_external_data(df_test)


# In[1036]:


test_dates_encoding = _encode_dates(df_test[["date"]])
df_test = pd.concat([df_test, test_dates_encoding], axis=1) 


# In[1037]:


df_test["counter_name"] = encoder.transform(df_test[["counter_name"]])


# In[1032]:


y_pred = pipe.predict(df_test[selected_features])
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)


# In[ ]:




