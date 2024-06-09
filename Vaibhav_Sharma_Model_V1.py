#!/usr/bin/env python
# coding: utf-8

# # KNN_Model_For_Breast_Cancer_Diagnostic

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


getDataSet = pd.read_csv('D:/Durham College/2nd Semester/AI in Enterprise System/Assignment 2/breast+cancer+wisconsin+diagnostic/wdbc.data', header=None)


# In[3]:


print(getDataSet)


# ## Separate features and target variable

# In[10]:


X = getDataSet.iloc[:, 2:32]
y = getDataSet.iloc[:, 1]


# In[11]:


print(X)


# ## Splitting the Data Set into Train and Test Set

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# ## Applying Feature Scaling

# In[13]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[14]:


print(X_train)


# ## Training the K-NN model on the Training set

# In[15]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
knn.fit(X_train, y_train)


# ## Predicting the Test set results

# In[16]:


y_pred = knn.predict(X_test)
print(y_pred)


# ## Making the Confusion Matrix

# In[17]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

