#!/usr/bin/env python
# coding: utf-8

# # Task - 2
# # Unemployment Analysis with Python

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


data = pd.read_csv('D:/oasis task-2.csv')


# In[4]:


data.head()


# In[5]:


# Change column name for understanding
data.columns=['States','Date','Frequency',
              'Estimated Unemployment Rate',
             'Estimated Employed',
             'Estimated Labour Participation Rate',
             'Region','longitude','latitude']


# In[6]:


data.head()


# In[7]:


data.describe()


# In[8]:


# check if the dataset contains missing values or not
print(data.isnull().sum())


# In[19]:


# Correlation between the feature of this Dataset
import seaborn as sns
sns.set(style='whitegrid')

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,10))
sns.heatmap(data.corr())
plt.show()


# In[21]:


#visualize the data to analyze the unemployment rate. estimated number of employees according to different regions of India:
data.columns= ["States", "Date", "Frequency",
               "Estimated Unemployment Rate", "Estimated Employed", 
               "Estimated Labour Participation Rate", "Region",
               "longitude","latitude"]
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Employed", hue="Region", data=data)
plt.show()


# In[22]:


#nemployment rate according to different regions of
plt.figure(figsize=(10, 8))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate", hue="Region", data=data)
plt.show()


# In[25]:


#create a dashboard to analyze the unemployment rate of each Indian state by region
unemploment = data[["States", "Region", "Estimated Unemployment Rate"]]
figure = px.sunburst (unemploment, path=["Region", "States"], 
                    width=700, height=700, color_continuous_scale="RdY1Gn",
                    values="Estimated Unemployment Rate",
                    title="Unemployment Rate in India")
figure.show()


# In[26]:


sns.pairplot(data)


# In[27]:


data.describe()


# In[28]:


X = data[['Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate',
          'longitude', 'latitude']]

y = data['Estimated Employed']


# In[69]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
X_train


# In[52]:


from sklearn.linear_model import LinearRegression


# In[54]:


lm = LinearRegression()


# In[63]:


#fit the model inside it 
lm.fit(X_train, y_train)


# In[64]:


#Evaluating Model
coeff_data = pd.DataFrame(lm.coef_,X.columns, columns = ['Coefficient'])


# In[65]:


# This table is saying 
# if one unit is increase then area will income incerase by $21
coeff_data


# In[66]:


# Predict the Model
predictions = lm.predict(X_test)


# In[74]:


print("y_test:", y_test)
print("predictions:", predictions)


# In[73]:


# Assuming y_test and predictions are NumPy arrays or lists
print(len(y_test), len(predictions))

# If they are not the same length, you might need to align or truncate one of them
min_length = min(len(y_test), len(predictions))
y_test = y_test[:min_length]
predictions = predictions[:min_length]

# Now, try plotting again
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()


# In[75]:


sns.distplot((y_test,predictions),bins=50)

