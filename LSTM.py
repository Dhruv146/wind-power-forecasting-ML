#!/usr/bin/env python
# coding: utf-8

# # Wind Energy Analysis and Prediction using LSTM

# In[70]:


#importing libraries
from pandas import read_csv
from pandas import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler  


# In[71]:


from numpy.random import randn
import os
import random
import time


# In[72]:


# load dataset
data = pd.read_csv("T1.csv")
data.head()


# In[73]:


data.describe()


# In[74]:


#correlation between the values
corr = data.corr()
plt.figure(figsize=(10, 8))

ax = sns.heatmap(corr, vmin = -1, vmax = 1, annot = True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
corr


# In[75]:


sns.pairplot(data)


# In[76]:


#msno.matrix(data)
# Spliting the date time in year, month, days, hours and minutes
data['Year']=data['Date/Time'].apply(lambda x: time.strptime(x,"%d %m %Y %H:%M")[0])
data['Month']=data['Date/Time'].apply(lambda x: time.strptime(x,"%d %m %Y %H:%M")[1])
data['Day']=data['Date/Time'].apply(lambda x: time.strptime(x,"%d %m %Y %H:%M")[2])
data['Time_Hours']=data['Date/Time'].apply(lambda x: time.strptime(x,"%d %m %Y %H:%M")[3])
data['Time_Minutes']=data['Date/Time'].apply(lambda x: time.strptime(x,"%d %m %Y %H:%M")[4])
data.head(10)


# In[77]:


#plt.subplots(figsize=(16, 16))
#sns.heatmap(data.corr(), annot=True, square=True)
#plt.show()
data["Date/Time"] = pd.to_datetime(data["Date/Time"], format = "%d %m %Y %H:%M", errors = "coerce")
data.head()


# In[78]:


data.info()


# In[79]:


#df_sel= data[['Unnamed: 0','ActivePower','WindSpeed']]
#data.head()


# In[80]:


data.describe()


# In[81]:


data.isnull().sum()


# In[82]:


#train, test = data_X[0:-144],data_X[-144:]

#history = [x for x in train]
# predictions = []
# for i in range(len(test)):

# 	predictions.append(history[-144])
# 	# observation
# 	history.append(test[i])
# # report performance
# rmse = sqrt(mean_squared_error(test, predictions))
# print('RMSE: %.3f' % rmse)

# plt.figure(figsize=(15, 8))
# plt.title('Predicted vs Actual Power Baseline')
# plt.plot(test, color='C0', marker='o', label='Actual Power')
# plt.plot(predictions, color='C1', marker='o', label='Predicted Power')
# plt.legend()
# plt.savefig('example.png')
# plt.show()


# In[83]:


def forecast_accuracy(forecast, actual):
    forecast = np.array(forecast)
    actual = np.array(actual)
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            'corr':corr, 'minmax':minmax})


# In[84]:


#using the MachineLearningMastery formula for splitting up the dataset to predictors and target
#reference: https://towardsdatascience.com/single-and-multi-step-temperature-time-series-forecasting-for-vilnius-using-lstm-deep-learning-b9719a0009de
def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    """
    A method to create X and Y matrix from a time series array for the training of 
    deep learning models 
    """
    # Extracting the number of features that are passed from the array 
    n_features = ts.shape[1]
    
    # Creating placeholder lists
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    # Reshaping the X array to an LSTM input shape 
    X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y


# In[85]:


# Number of lags (steps back in 10min intervals) to use for models to predict the next observation
lag = 360
# Steps in future to forecast (steps in 10min intervals) specifies the number of future time steps 
n_ahead = 144
# ratio of observations for training from total series
train_share = 0.8
# training epochs
epochs = 20
# Batch size , which is the number of samples of lags
batch_size = 256  #256
# Learning rate
lr = 0.001     #0.02
# The features for the modeling 
feat_final = ['Wind Speed (m/s)','Theoretical_Power_Curve (KWh)' ,'Wind Direction (°)','LV ActivePower (kW)']


# In[86]:


# Subseting only the needed columns 
ts = data[feat_final]


# In[87]:


#Scaling data between 0 and 1
object_ = StandardScaler()  
ts_scaled=object_.fit_transform(ts)  

# ts_scaled = scaler.transform(ts)


# In[88]:


# Creating the X and Y for training, the formula is set up to assume the target Y is the left most column = target_index=0
X,Y = create_X_Y(ts_scaled, lag=lag, n_ahead=n_ahead)


# In[89]:


# Spliting into train and test sets 
Xtrain, Ytrain = X[0:int(X.shape[0] * train_share)], Y[0:int(X.shape[0] * train_share)]
Xtest, Ytest = X[int(X.shape[0] * train_share):], Y[int(X.shape[0] * train_share):]
#print(Ytest)


# In[90]:


#Neural Network Model configuration, this is a Vanilla LSTM model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(16, activation='relu', return_sequences=False))
#model.add(tf.keras.layers.CuDNNLSTM(32, return_sequences=False)) you can try to use the 10x faster GPU accelerated CuDNNLSTM instaed of the Vanilla LSTM above, but do not forget to set up the notebook accelerator to "GPU"
model.add(tf.keras.layers.Dense(144))

#set up early stop function to stop training when val_loss difference is higher than 0.001
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, min_delta=0.001)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae', metrics='mae')


# In[ ]:


#Train model on train data and use test data for validation
#If the model does not converge accurately, you need check if it is a input data quality issue, introduce a dropout layer, or you can try adjusting the number of hidden nodes
#history = model.fit(Xtrain, Ytrain,epochs=epochs, validation_data=(Xtest, Ytest), shuffle=False, callbacks=[early_stopping])
history = model.fit(Xtrain, Ytrain,epochs=epochs, validation_data=(Xtest, Ytest), shuffle=False)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label = 'Validation loss')
plt.legend()


# In[ ]:


#predict based on test data
yhat = model.predict(Xtest)
#print(yhat)


# In[ ]:


# Creating the predictions date range
days = time_df.values[-len(yhat):-len(yhat) + n_ahead]
days_df = pd.DataFrame(days)
print(days_df)


# In[ ]:


#prepare resulting series for inverse scaling transformation
#pay attention we will select only the first prediction we have made, therefore [0] used to select this window (we have generated multiple prediction sequences of 144 steps ahead, starting from each interval step in the test dataset)
pred_n_ahead = pd.DataFrame(yhat[0])
actual_n_ahead = pd.DataFrame(Ytest[0])
print(pred_n_ahead)
#repeat the column series 2 times, to make shape compatible for scale inversion
pr_p = pd.concat([pred_n_ahead], axis=1)
print("yes",pr_p)
ac_p = pd.concat([actual_n_ahead], axis=1)
print(ac_p.shape)
#print(pred_n_ahead)


# In[ ]:


# #inverse scale tranform the series back to kiloWatts of power
# print(pr_)
# pr_p = pd.DataFrame(scaler.inverse_transform(pr_p))
# ac_p = pd.DataFrame(scaler.inverse_transform(ac_p))

# #rename columns
pr_p = pr_p.rename(columns={0:'PredPower'})
ac_p = ac_p.rename(columns={0:'ActualPower'})
print(ac_p)
# #concatenate together into one dataframe and set index
df_final = pd.concat([days_df, pr_p['PredPower'], ac_p['ActualPower']], axis=1).set_index(0)
df_final


# In[ ]:


plt.figure(figsize=(15, 8))
plt.plot(df_final.index, df_final.ActualPower, color='C0', marker='o', label='Actual Power')
plt.plot(df_final.index, df_final.PredPower, color='C1', marker='o', label='Predicted Power', alpha=0.6)
plt.title('Predicted vs Actual Power')
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.legend()
plt.savefig('forecast_example.png')
plt.show


# In[ ]:


forecast_accuracy(df_final['PredPower'], df_final['ActualPower'])


# In[ ]:


# ##residuals summary stats
print(df_final.head())
print(df_final['PredPower'].head(5))
df_final['Residuals'] =  df_final['PredPower'] - df_final['ActualPower']
print(df_final.shape)
df_final['Residuals'].describe()


# In[ ]:


#residuals histogram shows the bias (Mean Error)
df_final['Residuals'].hist(color = ('C0'))
plt.title('Residuals histrogram plot')
plt.xlabel('Bins')
plt.ylabel('Occurance count')


# In[ ]:


#density plot shows the bias (Mean Error)
df_final['Residuals'].plot(kind='kde')


# In[ ]:


#Q-Q plot showing normal distrubition of data
from statsmodels.graphics.gofplots import qqplot
qqplot(df_final['Residuals'])


# In[ ]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df_final['Residuals'])
plt.show()


# In[ ]:




