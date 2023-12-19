#!/usr/bin/env python
# coding: utf-8

# # Oasis Infobyte Internship, December-2023

# # Iris Flowers Classification ML Project

# # Import Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#supress warnings
import warnings
warnings.filterwarnings('ignore')


# # Import Dataset

# In[3]:


data = pd.read_csv('D:/oasis/iris.csv')


# In[4]:


data


# # Data Exploration

# In[5]:


# check shape of Data
data.shape


# In[6]:


# check basic information of Data
data.info()


# In[7]:


# check stastical summary of Data
data.describe()


# In[8]:


# check null values
data.isnull().sum()


# NO NUll values in dataset

# In[9]:


print('unique number of values in dataset Species:',data['Species'].unique())
print('unique Species in iris dataset:',data['Species'].unique())


# There is 3 species in iris dataset that is [ 'iris-sentosa' 'iris-veriscolor' 'iris-virginica' ]

# # Exploratory Data Analysis
# 
# # Data Visualization

# In[10]:


sns.pairplot (data, hue = 'Species', markers = 'x')
plt.show()


# It shows that Iris-Setosa it separated from both other species in all features

# In[11]:


sns.scatterplot(x='SepalLengthCm', y='PetalLengthCm', data=data, hue='Species')
plt.show()


# In[12]:


sns.scatterplot(x='SepalWidthCm', y='PetalWidthCm', data=data, hue='Species')
plt.show()


# In[13]:


# check correlation in Dataset
data.corr()


# In[14]:


# Use Heatmap to see Correlation
plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True, cmap='Oranges_r')
plt.show()


# In the above Heatmap we see petal length and petal width is highly Correlated

# In[15]:


# Check value counts
data['Species'].value_counts().plot(kind='pie',autopct = '%1.1f%%', shadow = True, figsize = (5,5))
plt.title('Percentage values in each Species', fontsize = 12, c = 'g')
plt.ylabel('', fontsize=10,c='r')
plt.show()


# We can see, all species has eaual values in a dataset

# # Scatterplot for sepal length and sepal width

# In[16]:


sns.scatterplot(x=data['SepalLengthCm'], y=data['SepalWidthCm'], hue=data['Species'])
plt.show()


# In[17]:


sns.scatterplot(x=data['PetalLengthCm'], y=data['PetalWidthCm'], hue=data['Species'])
plt.show()


# In[18]:


plt.figure(figsize=(15, 5))

plt.subplot(2, 2, 1)
sns.barplot(x="Species", y="SepalLengthCm", data=data, palette="Spectral")
plt.title("Bar plot Sepal LengthCm Vs Species")

plt.subplot(2, 2, 2)
sns.boxplot(x="Species", y="SepalLengthCm", data=data, palette="Spectral")
plt.title("Box plot Sepal LengthCm Vs Species")

plt.subplot(2, 2, 3)
sns.barplot(x="Species", y="SepalWidthCm", data=data, palette="Spectral")
plt.title("Bar plot Sepal WidthCm Vs Species")

plt.subplot(2, 2, 4)
sns.boxplot(x="Species", y="SepalWidthCm", data=data, palette="Spectral")
plt.title("Box plot Sepal WidthCm Vs Species")

plt.tight_layout() 
plt.show()


# In[19]:


plt.figure(figsize=(15, 5))

plt.subplot(2, 2, 1)
sns.barplot(x="Species", y="SepalLengthCm", data=data, palette="coolwarm")
plt.title("Bar plot Sepal LengthCm Vs Species")

plt.subplot(2, 2, 2)
sns.boxplot(x="Species", y="SepalLengthCm", data=data, palette="coolwarm")
plt.title("Box plot Sepal LengthCm Vs Species")

plt.subplot(2, 2, 3)
sns.barplot(x="Species", y="SepalWidthCm", data=data, palette="coolwarm")
plt.title("Bar plot Sepal WidthCm Vs Species")

plt.subplot(2, 2, 4)
sns.boxplot(x="Species", y="SepalWidthCm", data=data, palette="coolwarm")
plt.title("Box plot Sepal WidthCm Vs Species")

plt.tight_layout()
plt.show()


# # Data Cleaning

# In[20]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Species'] = le.fit_transform(data['Species'])
data.head()


# In[22]:


data['Species'].unique()


# In[24]:


X = data.iloc[:,[0,1,2,3]]
X.head()


# In[25]:


y = data.iloc[:,-1]
y.head()


# In[26]:


print(X.shape)
print(y.shape)


# # Model Building
# 
# # Supervised Machine Learning
# 
# # Split data into Training and Testing set

# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


# In[29]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Logistic Regression

# In[36]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(X_train, y_train)
print("Logistic regression successfully implemented")
y_pred = lr.predict(X_test)
#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:-")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy is:-", accuracy*100)
print("Classification Report:-")
print(classification_report(y_test,y_pred))


# In[41]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print("Rndom Forest Classifier successfully Implimented")
y_pred = rfc.predict(X_test)
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ") 
print(cm)
#accuracy test
accuracy = accuracy_score(y_test,y_pred)
print("accuracy:- ", accuracy*100)
print("Classification Report:-")
print( classification_report(y_test, y_pred))


# # Decision Tree

# In[43]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
print("Decision Tree Algorithm is successfully implimented.")
y_pred = dtree.predict(X_test)
#confusion matrix
cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix: ")
print(cm)
#accuracy test
accuracy = accuracy_score(y_test,y_pred)
print("accuracy:- ", accuracy*100)
print("Classification Report:-")
print( classification_report(y_test, y_pred))


# In[44]:


from sklearn.tree import plot_tree


# In[45]:


# for visualziing the Decision Tree
feature= ['SepalLengthCm','SepalwidthCm','PetalLengthCm', 'PetalwidthCm']
classes=['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']
plt.figure(figsize=(10,10))
plot_tree(dtree, feature_names = feature, class_names = classes, filled = True);


# # Support Vector Machine

# In[46]:


from sklearn.svm import SVC
svc= SVC()
svc.fit(X_train, y_train)
print("Support vactor classifier is successfully implemented")
y_pred = svc.predict(X_test)
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)
#accuracy test
accuracy = accuracy_score(y_test,y_pred)
print("accuracy:- ", accuracy*100)
print("Classification Report:-")
print( classification_report(y_test, y_pred))


# # K - NN Classifier

# In[48]:


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors= 7)
knn.fit(X_train, y_train)
print("K-Nearest Neighbors classifier is successfully implemented")
y_pred = knn.predict(X_test)

#confusion matrix
cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:- ")
print(cm)

#accuracy test
accuracy = accuracy_score(y_test,y_pred)
print("accuracy: ", accuracy*100)
print("Classification Report:-")
print( classification_report(y_test, y_pred))


# # Naive Bayes

# In[52]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train) 
print("Naive Bayes is successfully implemented")

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print("Confusion Matrix: ")
print(cm)

#Accuracy test
accuracy = accuracy_score(y_test, y_pred)
print("accuracy:-", accuracy*100)
print("Classification Report:-")
print( classification_report(y_test, y_pred))


# # Result
#1.Accuracy of Logistic Regression :- 100%
#2.Accuracy of  Random Forest Classifier :- 100%
#3.Accuracy of Decission Tree :- 96.66%
#4.Accuracy of Support vector Machine :- 100%
#5.Accuracy of K-NN Classifier :- 100%
#6.Accuracy of Naive Bayes :- 100%
# # Test Model

# In[55]:


input_data=(4.9,3.0,1.4,0.2)

#changing the input data to a numpy array 
input_data_as_nparray = np.asarray(input_data)

#reshape the data as we are predicting the label for only the instance 
input_data_reshaped = input_data_as_nparray.reshape(1,-1)
prediction = dtree.predict(input_data_reshaped)
print("The category is", prediction)

