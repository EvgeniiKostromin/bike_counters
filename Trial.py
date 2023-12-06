#!/usr/bin/env python
# coding: utf-8

# In[26]:


from pathlib import Path
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[27]:


df_train = pd.read_parquet(Path("data") / "train.parquet")


# In[28]:


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


# In[29]:


def _merge_external_data(X):
    file_path = Path('data') / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    
    X['date'] = pd.to_datetime(X['date']).astype('datetime64[ns]')
    df_ext['date'] = pd.to_datetime(df_ext['date']).astype('datetime64[ns]')
    
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t", 'pres', 'raf10', 'u', 'vv', 'tend24', 'cod_tend', 'rr3', 'rr12']].sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


# In[30]:


df_train = _merge_external_data(df_train)


# In[31]:


df_train['counter_age_days'] = df_train['date'] - df_train['counter_installation_date']
df_train['counter_age_days'] = df_train['counter_age_days'].dt.days


# In[32]:


X_dates_encoding = _encode_dates(df_train[["date"]])
df_train = pd.concat([df_train, X_dates_encoding], axis=1) 


# In[33]:


df_train['rr3'] = np.exp(-2.4 * df_train['rr3']) 
df_train['rr12'] = np.exp(-0.2 * df_train['rr12']) 


# In[34]:


y_train = df_train['log_bike_count']
X_train = df_train.drop('log_bike_count', axis=1)


# In[35]:


from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

date_cols = ["year", "month", 'day', 'weekday', 'hour']
scaling_columns = ['counter_age_days', 'raf10', 't', 'pres', 'u', 'vv', 'tend24', 'latitude', 'rr3', 'rr12']
categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = ["counter_name", "site_name", 't', 'pres', 'cod_tend']

imputer = SimpleImputer(strategy='mean')

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
        ("cat", categorical_encoder, categorical_cols),
        ('standard-scaler', StandardScaler(), scaling_columns),
        
    ]
)

regressor = Ridge(alpha=10)

pipe = make_pipeline(preprocessor, imputer, regressor)
pipe.fit(X_train, y_train)


# In[37]:


df_test = pd.read_parquet(Path("data") / "final_test.parquet")


# In[38]:


df_test = _merge_external_data(df_test)


# In[39]:


test_dates_encoding = _encode_dates(df_test[["date"]])
df_test = pd.concat([df_test, test_dates_encoding], axis=1) 


# In[40]:


df_test['counter_age_days'] = df_test['date'] - df_test['counter_installation_date']
df_test['counter_age_days'] = df_test['counter_age_days'].dt.days


# In[41]:


df_test ['rr3'] = df_test ['rr3'].fillna(df_test ['rr3'].mean())
df_test ['rr12'] = df_test ['rr12'].fillna(df_test ['rr3'].mean())


# In[42]:


df_test['rr3'] = np.exp(-2.4 * df_test['rr3']) 
df_test['rr12'] = np.exp(-0.2 * df_test['rr12']) 


# In[43]:


y_pred = pipe.predict(df_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)

