#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[2]:


df_train = pd.read_parquet(Path("data") / "train.parquet")


# In[3]:


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


# In[4]:


def _merge_external_data(X):
    file_path = Path('data') / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X['date'] = pd.to_datetime(X['date']).astype('datetime64[ns]')
    df_ext['date'] = pd.to_datetime(df_ext['date']).astype('datetime64[ns]')

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[[
            "date", "t", 'u'
        ]].sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


# In[5]:


df_train = _merge_external_data(df_train)


# In[6]:


X_dates_encoding = _encode_dates(df_train[["date"]])
df_train = pd.concat([df_train, X_dates_encoding], axis=1)


# In[7]:


y_train = df_train['log_bike_count']
X_train = df_train.drop('log_bike_count', axis=1)


# In[8]:


from sklearn.preprocessing import OrdinalEncoder
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


categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = ["counter_name"]
passthrough_cols = ['latitude', 't', 'u', 'month', 'weekday', 'hour', 'day']

preprocessor = ColumnTransformer(
    [
        ("cat", categorical_encoder, categorical_cols),
        ('passthrough', 'passthrough', passthrough_cols)
    ]
)

regressor = XGBRegressor(
    learning_rate=0.1, max_depth=9, n_estimators=980, gamma=0,
    min_child_weight=7, reg_alpha=0.2, colsample_bytree=0.85,
    reg_lambda=6, max_delta_step=1
)

pipe = make_pipeline(preprocessor, regressor)
pipe.fit(X_train, y_train)


# In[9]:


df_test = pd.read_parquet(Path("data") / "final_test.parquet")


# In[10]:


df_test = _merge_external_data(df_test)


# In[11]:


test_dates_encoding = _encode_dates(df_test[["date"]])
df_test = pd.concat([df_test, test_dates_encoding], axis=1)


# In[12]:


y_pred = pipe.predict(df_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)


# In[ ]:




