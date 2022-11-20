#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


ccdata=pd.read_csv('C:\pandas\creditcard.zip')
ccdata


# In[3]:


ccdata.head()


# In[4]:


ccdata.tail()


# In[5]:


ccdata.info()


# In[6]:


ccdata.isnull().sum()


# In[7]:


ccdata['Class'].value_counts()


# In[8]:


genuine = ccdata[ccdata.Class == 0]
fraud = ccdata[ccdata.Class == 1]


# In[9]:


print(genuine.shape)


# In[10]:


print(fraud.shape)


# In[11]:


genuine.Amount.describe()


# In[12]:


fraud.Amount.describe()


# In[13]:


ccdata.groupby('Class').mean()


# In[14]:


genuine_sample = genuine.sample(n=492)


# In[15]:


new = pd.concat([genuine_sample,fraud] , axis = 0)
new.head()


# In[16]:


new.tail()


# In[17]:


new['Class'].value_counts()


# In[18]:


new.tail(495)


# In[20]:


X = new.drop(columns='Class' , axis =1)
Y = new['Class']


# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 , stratify = Y , random_state =2);


# In[24]:


X.shape , X_train.shape , X_test.shape


# In[25]:


model = LogisticRegression()


# In[26]:


model.fit(X_train,Y_train)


# In[31]:


X_train_Pr = model.predict(X_train)


# In[33]:


training_data_acc = accuracy_score(X_train_Pr,Y_train)


# In[35]:


training_data_acc


# In[36]:


X_test_Pr = model.predict(X_test)
testing_data_acc = accuracy_score(X_test_Pr , Y_test)


# In[37]:


testing_data_acc


# In[ ]:




