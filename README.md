
Clayfin

Fin Cast

Overview
A time series modeling approach (ARIMAX model) has been used in the Fin cast (Forecasting the users expenses and incomes). The order of the best model is automatically calculated by using the ADF test. The category wise forecast is for the next 180 day from the user's last transaction. 
Goals
To predict the users expenses and incomes based on their transactions by using machine learning.
Automatic model adjustment to get better accuracy.
What is ARIMAX
An Autoregressive Integrated Moving Average with Explanatory Variable (ARIMAX) model can be viewed as a multiple regression model with one or more autoregressive (AR) terms and/or one or more moving average (MA) terms. This method is suitable for forecasting when data is stationary/non stationary, and multivariate with any type of data pattern, i.e., level/trend /seasonality/cyclicity.
Data Requirement
The time series data should be equal intervals. Here, we are taking summation of category wise transaction amounts into hourly and the predicted data are also hourly. 
The category wise transaction should be a minimum of 30 days and minimum of 100 days of transaction including zero amount transaction days. 


Data Preparation For ARIMAX
The category wise both expense and income transactions are taken from the mysql database and converted into hourly. The month  days and week days are calculated from the date which are used as the explanatory variable for the model.
The category wise next 180 days from the last day of transactions are created for the model. 
The machine learning model is exception for the following income categories
Salary
Rent
The salary and rent amount are predicted for next 180 days are the same as on their last month transaction amount and the same day of the month.
The Machine Learning Model
The ARIMAX model is created for each user and their categories. The model parameters are tuning automatically for each model to get good accuracy. 
The Output 
The predicted data are arranged and moved to the mysql database 


import time
import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import mysql.connector as sql
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pmdarima
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings
from datetime import datetime
from datetime import time
import datetime

df = pd.read_csv('fincast_inut.csv')
df['transaction_timestamp'] = df['transaction_timestamp'].dt.date


#data = df[(df.type == 'CR')]
data1 = df[['transaction_timestamp', 'customer', 'category', 'transaction_amount', 'type']]

df1 = data1.groupby(['transaction_timestamp', 'customer','type', 'category'])['transaction_amount'].sum().reset_index()
tdf1 = df1.groupby(['customer','type','category']).agg(count=('customer', 'count'))
tdf2 = pd.merge(df1, tdf1, on=['customer','type','category'], how='left')

df_train = tdf2.sort_values(["customer",'type',"category","transaction_timestamp"],ascending=False)
recent_trans = df_train.drop_duplicates(subset = ['customer', 'type', 'category'], keep='first')
recent_trans.rename({'transaction_timestamp':'recentdate'}, axis = 1, inplace = True)
df2 = tdf2.sort_values(["customer",'type',"transaction_timestamp","category"],ascending=True)
first_trans = df2.drop_duplicates(subset = ['customer','type', 'category'], keep='first')
first_trans = first_trans.drop(['transaction_amount', 'count'], axis = 1)
first_trans.rename({'transaction_timestamp':'firstdate'}, axis = 1, inplace = True)

unique_trans = pd.merge(recent_trans, first_trans, on=['customer', 'type', 'category'], how='left')

from datetime import timedelta, date
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

df_traincol = df1.columns
df_train1 = pd.DataFrame(columns = df_traincol)
df_train1 = df_train1.drop(['transaction_amount'], axis = 1)
        
dfdate = []
dfcus = []
dftype = []
dfcat = []
count = []
for ti in range(0, len(unique_trans)):
    rd = unique_trans.iloc[ti]['recentdate']
    fd = unique_trans.iloc[ti]['firstdate']
    dc = unique_trans.iloc[ti]['customer']
    ty = unique_trans.iloc[ti]['type']
    cat = unique_trans.iloc[ti]['category']
    co = unique_trans.iloc[ti]['count']
    #day_delta = datetime.timedelta(days=1)
    for dt in daterange(fd, rd):
        dfdate.append(dt)
        dfcus.append(dc)
        dftype.append(ty)
        dfcat.append(cat)
        count.append(co)
df_train1['customer'] = dfcus
df_train1['transaction_timestamp'] = dfdate
df_train1['type'] = dftype
df_train1['category'] = dfcat
df_train1['count'] = count

train = pd.merge(df_train1, df_train, on=['customer', 'type', 'transaction_timestamp', 'category', 'count'], how='left')
train = train.fillna(0)
trainm = train[(train['type'] != 'CR') | (train['category'] != 10) & (train['category'] != 7)]
train1 = trainm[trainm['count'] > 0]
train1 = train1.drop(['count'], axis = 1)
#finding overall rows in the model customer and category wise
traingr = train1.groupby(['customer','type','category']).size().reset_index(name='count')
trainme = pd.merge(train1, traingr, on=['customer', 'type', 'category'], how='left')
training = trainme[trainme['count'] > 100] 
training = training.drop(['count'], axis=1) 
tempcol = df1.columns
test = pd.DataFrame(columns = tempcol)
cus = []
date = []
tcat = []
ttype = []
for ti in range(0, len(unique_trans)):
    fd = unique_trans.iloc[ti]['firstdate']
    dc = unique_trans.iloc[ti]['customer']
    ty = unique_trans.iloc[ti]['type']
    cat = unique_trans.iloc[ti]['category']
    day_delta = datetime.timedelta(days=1)
    sd = datetime.datetime.now()
    ed = sd + 183*day_delta
    for dt in daterange(sd, ed):
        cus.append(dc)
        date.append(dt)
        tcat.append(cat)
        ttype.append(ty)
test['customer'] = cus
test['type'] = ttype
test['transaction_timestamp'] = date
test['category'] = tcat

df_test = test[(test['type'] != 'CR') | (test['category'] != 10) & (test['category'] != 7)]
df_test =  pd.merge(df_test, traingr, on=['customer', 'type', 'category'], how='left')
df_test = df_test[df_test['count'] > 0] 
df_test = df_test.drop(['count'], axis=1)

training['transaction_timestamp'] =  pd.to_datetime(training['transaction_timestamp'], infer_datetime_format=True)
training['transaction_timestamp'] = pd.to_datetime(training['transaction_timestamp'])
training = training.set_index('transaction_timestamp')
df_test['transaction_timestamp'] = pd.to_datetime(df_test['transaction_timestamp'])
df_test = df_test.set_index('transaction_timestamp')

day = training.index.day
dummy_day = pd.get_dummies(day)
dummy_day.columns = ['day-%s' % m for m in dummy_day.columns]
dummy_day.index = training.index

weekday = training.index.weekday
dummy_weekday = pd.get_dummies(weekday)
dummy_weekday.columns = ['Wday-%s' % m for m in dummy_weekday.columns]
dummy_weekday.index = training.index

training = pd.concat([training, dummy_day, dummy_weekday], axis=1)
training.dropna(inplace=True)
training.head()

import datetime

df_test_ex = df_test.copy()
day = df_test_ex.index.day
dummy_day = pd.get_dummies(day)
dummy_day.columns = ['day-%s' % m for m in dummy_day.columns]
dummy_day.index = df_test_ex.index

weekday = df_test_ex.index.weekday
dummy_weekday = pd.get_dummies(weekday)
dummy_weekday.columns = ['Wday-%s' % m for m in dummy_weekday.columns]
dummy_weekday.index = df_test_ex.index

df_test_ex = pd.concat([df_test_ex, dummy_day, dummy_weekday], axis=1)
#df_test_ex.dropna(inplace=True)
df_test_ex.head()

arimax_results = df_test.reset_index()
arimax_results['transaction_amount'] = 0
arimax_results
''''''''''
tramo = df1.copy()
tramo['transaction_timestamp'] = pd.to_datetime(tramo.transaction_timestamp)
#tramo['day'] = tramo['transaction_timestamp'].dt.day
tramo = tramo.sort_values(["customer","transaction_timestamp","category"],ascending=True)
#tramo3 = tramo.drop_duplicates(subset = ['customer', 'category', 'day'], keep='first')
tramo1 = tramo.set_index('transaction_timestamp')
tramo2 = tramo1.last("30D")

for i in tramo1['customer'].unique():
    print(i)
    for ca in tramo1['category'].unique():
        print(ca)
        last = tramo1.loc[(tramo1['customer'] == i) & (tramo1['category'] == ca)].last('30D')
        
'''''''''''


trainingDB = training[training['type'] == "DB"].drop(['type'], axis = 1)
df_test_exDB = df_test_ex[df_test_ex['type'] == "DB"].drop(['type'], axis = 1)
df_testDB = df_test[df_test['type'] == "DB"].drop(['type'], axis = 1)

trainingCR = training[training['type'] == "CR"].drop(['type'], axis = 1)
df_test_exCR = df_test_ex[df_test_ex['type'] == "CR"].drop(['type'], axis = 1)
df_testCR = df_test[df_test['type'] == "CR"].drop(['type'], axis = 1)

warnings.filterwarnings("ignore") # specify to ignore warning messages
temp = pd.DataFrame()
def autoARIMAX(trainingDB, df_test_exDB, df_testDB, temp): 
    for s in trainingDB['customer'].unique():
        print(s)
        for c in trainingDB['category'].unique():
            print(c)
            endog = trainingDB.loc[(trainingDB['customer'] == s) & (trainingDB['category'] == c), 'transaction_amount']
            exog = trainingDB.loc[(trainingDB['customer'] == s) & (trainingDB['category'] == c)].drop(['customer', 'category', 'transaction_amount'], axis=1)
            print(len(endog))
            print(len(exog))
            if len(endog) !=0:
                arimax = pmdarima.arima.auto_arima(endog, exogenous=exog,
                                       start_p=0, d=0, start_q=0,
                                       test='adf',
                                       max_p=5, max_d=5, max_q=5, start_P=0, D=None, start_Q=0, m=1,
                                       max_P=5, max_D=5, max_Q=5, seasonal=True, trace=True,
                                       error_action='warn',  
                                       suppress_warnings=True, 
                                       stepwise=True)
                exog_test = df_test_exDB.loc[(df_test_exDB['customer'] == s) & (df_test_exDB['category'] == c)].drop(['customer', 'category', 'transaction_amount'], axis=1)
                n_periods = 184
                fitted, confint = arimax.predict(n_periods=n_periods, 
                                              exogenous=exog_test, 
                                              return_conf_int=True)
                #fitted_series = pd.Series(fitted)
                empty = df_testDB.loc[(df_testDB['customer'] == s) & (df_testDB['category'] == c)]
                empty['transaction_amount'] = fitted
                empty['transaction_amount'][empty['transaction_amount'] < 0] = 0
                temp = pd.concat([empty, temp])
    return temp
        
tempDB = autoARIMAX(trainingDB, df_test_exDB, df_testDB, temp)
tempDB['txn_type'] = 'DB'
tempCR = autoARIMAX(trainingCR, df_test_exCR, df_testCR, temp)
tempCR['txn_type'] = 'CR'
tempfinal = pd.concat([tempDB, tempCR])
temp_re = tempfinal.replace(0, np.nan)
temp_re.reset_index(level=0, inplace=True)
temp_re.rename({'index':'transaction_timestamp'}, axis = 1, inplace = True)


#creating 31 days of last stable income categories transaction
cunique_trans = unique_trans[(unique_trans['category'] == 10) | (unique_trans['category'] == 7)]
cunique_trans['recentdate'] = pd.to_datetime(cunique_trans['recentdate'])
cunique_trans['day'] = cunique_trans['recentdate'].dt.day
cunique_trans['recentdate'] = cunique_trans['recentdate'].dt.date
ctrain = cunique_trans[['day', 'customer', 'type', 'category','transaction_amount']]


#Creating next 180 days of stable income categories
ctest = test[(test['type'] == 'CR')]
ctest = ctest[(ctest['category'] == 10) | (ctest['category'] == 7)]
ctest['transaction_timestamp'] = pd.to_datetime(ctest.transaction_timestamp)
ctest['day'] = ctest['transaction_timestamp'].dt.day
ctest1 = ctest.drop(['transaction_amount'], axis=1)

#merging the amount to 180 days of each income categories based on the day
ctest2 =  pd.merge(ctest1, ctrain, on=['day', 'customer', 'type', 'category'], how='left')
ctest3 = ctest2.drop(['day'], axis=1)
ctest3.rename(columns={'type':'txn_type'}, inplace=True)

final = pd.concat([temp_re, ctest3])
final = final[final['transaction_amount'].notna()]
final['month'] = final['transaction_timestamp'].dt.month
final['year'] = final['transaction_timestamp'].dt.year
final['quarter'] = final['transaction_timestamp'].dt.quarter
final.rename(columns={'transaction_timestamp':'txn_date', 'customer':'customer_id', 'transaction_amount':'txn_amount'}, inplace=True)
final['txn_date'] = final['txn_date'].apply(lambda x: x.strftime('%Y-%m-%d'))

final.info()




