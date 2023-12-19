#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv('D:/oasis task-3.csv')
data


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.isnull().sum()


# In[7]:


data = data.dropna()


# In[8]:


data = data.drop_duplicates()


# In[9]:


plt.style.use('dark_background')
sns.set_palette('dark')
sns.histplot(data['Selling_Price'])
plt.title('Distribution of Car Prices', color="white")
plt.xlabel('Selling_Price', color='white')
plt.ylabel('Count', color="white")
plt.xticks(color="white")
plt.yticks(color="white")
plt.show()


# In[10]:


plt.style.use('dark_background')
sns.set_palette('dark')
sns.histplot(data['Present_Price'])
plt.title('Distribution of Car Prices', color="white")
plt.xlabel('Present_Price', color='white')
plt.ylabel('Count', color="white")
plt.xticks(color="white")
plt.yticks(color="white")
plt.show()


# In[11]:


# Correlation Heatmap
numeric_features = ['Car_Name','Year','Selling_Price','Present_Price',
                    'Driven_kms','Fuel_Type','Selling_type','Transmission','Owner']
correlation_matrix = data[numeric_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm') 
plt.title('Correlation Heatmap', color='white')
plt.xticks(color="white")
plt.yticks(color='white')
plt.show()


# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# In[25]:


feature_cols = ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation',
                  'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
                  'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm',
                  'citympg', 'highwaympg']

target_col = 'price'
x=data.corr()
x


# In[30]:


from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
import seaborn as sns
label_encoder = LabelEncoder()
for col in x.columns:
    if x[col].dtype =='object':
        x[col] = label_encoder.fit_transform(x[col])


# # Splitting data

# In[32]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# # Fitting the Model

# In[35]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# # Evaluating the Model

# In[38]:


import numpy as np
from sklearn.metrics import mean_squared_error
predictions = model.predict(x_test)


# In[48]:


mse = mean_squared_error
print('Mean Squared Error:')


# In[64]:


import pandas as pd

new_car_data = [[3, 'gas', 'std', 'two', 'sedan', 'fwd', 'front", 100.0, 180.0, 68.0, 56.0, 2500, ohc', 'four',
                 120, 'mpfi', 3.50, 2.80, 8.5, 110, 5500, 30, 38]]
feature_cols=['num_doors', 'fuel_type', 'aspiration', 'body_style', 'drive_wheels', 'engine_location',
                               'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type', 'num_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'symboling']
dataset = pd.read_csv('D:/oasis task-3.csv')


# # Predictions

# In[67]:


data = {
    'Feature': [1, 2, 3, 4, 5],
    'Price': [10, 20, 30, 40, 50]
}

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))


plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Feature')
plt.ylabel('Price')
plt.show()

