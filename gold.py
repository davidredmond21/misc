#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ops

#get_ipython().run_line_magic("matplotlib", 'inline')

from sklearn.preprocessing import MinMaxScaler
import tensorflow
import tensorflow.keras
from keras_tqdm import TQDMNotebookCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras.backend as K

from keras_tqdm import TQDMNotebookCallback

# In[2]:


df = pd.read_csv('golds.csv')

# In[3]:


df.columns = ['date', 'price']
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df.head()

# In[4]:


df.plot(figsize=(15, 10))

# In[5]:


df.describe()

# In[6]:


split_date = pd.Timestamp('10-01-2016')

# In[7]:


train = df.loc[:split_date, ['price']]
test = df.loc[split_date:, ['price']]

# In[8]:


ax = train.plot()
test.plot(ax=ax, figsize=(15, 10))
plt.legend(['train', 'test'])

# In[9]:


sc = MinMaxScaler()

train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

# In[10]:


train_sc[:4]

# In[11]:


train[:4]

# In[12]:


X_train = train_sc[:-1]
y_train = train_sc[1:]

X_test = test_sc[:-1]
y_test = test_sc[1:]

# In[13]:


train_sc.shape

# In[14]:


y_train

# In[15]:


X_train_t = X_train[:, None]
X_test_t = X_test[:, None]

# In[16]:


X_train_t.shape

# In[17]:


K.clear_session()
model = Sequential()

model.add(LSTM(4, input_shape=(1, 1)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# In[18]:


model.fit(X_train_t, y_train,
          epochs=10, batch_size=1, verbose=0,
          callbacks=[TQDMNotebookCallback(leave_inner=True)])

# In[19]:


y_pred = model.predict(X_test_t)
plt.figure(figsize=(15, 10))
plt.plot(y_test)
plt.plot(y_pred)
plt.legend(['real', 'predict'])

# In[20]:


sc.inverse_transform(y_pred)[-1]

# In[ ]:
