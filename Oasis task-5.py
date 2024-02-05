#!/usr/bin/env python
# coding: utf-8

# # Task - 5
# 
# # Sales Prediction Using Python

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('D:/sales.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.describe()


# In[6]:


data.info()


# # Dropping the Column

# In[7]:


data = data.drop('Unnamed: 0', axis=1)


# In[8]:


data


# In[9]:


x = data.iloc[:,0:-1]


# In[10]:


x


# In[11]:


y=data.iloc[:,-1]
y


# In[12]:


print(data.shape)


# # Train test split

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)


# In[15]:


x_train


# In[16]:


x_test


# In[17]:


y_train


# In[18]:


x_train=y_train.astype(int)
y_train=y_train.astype(int) 
x_test=x_test.astype(int) 
y_test-y_test.astype(int)


# In[19]:


import numpy as np
from sklearn.preprocessing import StandardScaler

x_train_numpy = x_train.to_numpy()
x_train_reshaped = x_train_numpy.reshape(-1, 1)

# Create the StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_reshaped)
x_train_scaled_series = pd.Series(x_train_scaled.flatten(), index=x_train.index)


# # Apllying Linear Regression

# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


lr = LinearRegression()


# In[22]:


lr.fit(x_train_scaled,y_train)


# In[23]:


y_pred = lr.predict(x_train_scaled)


# In[24]:


y_pred


# In[25]:


import matplotlib.pyplot as plt
import numpy as np
y_test = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 140, 190, 240, 310])

plt.scatter(y_test, y_pred, color='blue', label='sales Prediction scatter plot')
plt.xlabel('TV')
plt.ylabel('Radio')
plt.title('Sales prediction Scatter Plot')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Approximate Prediction')
plt.legend()
plt.show()

