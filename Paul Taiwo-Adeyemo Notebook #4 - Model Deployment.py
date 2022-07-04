#!/usr/bin/env python
# coding: utf-8

# ### Capstone Project: Using Machine Learning to Reduce Human/Medical costs Associated with Diabetes.

# ##### Paul Taiwo-Adeyemo, July 2022, BrainStation

# ##### Notebook #4: Model Deployment

# In this notebook the Logistic Regression Model from Dataset 2 would be deployed. This deployment would input lifestyle choices to predict the incidence of diabetes.

# In[6]:


#import relevant packages
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import packages for Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# Ignore futurewarnings
import warnings
warnings.filterwarnings('ignore')
#Import necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[7]:


#load dataset
dataset2_df = pd.read_csv('dataset2_cleaned.csv')


# In[8]:


#define X and Y
X = dataset2_df.drop('Diabetes_binary', axis=1)
Y = dataset2_df['Diabetes_binary']


# In[9]:


#split into train and test 
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=3)


# In[10]:


#scale the data
scaler = MinMaxScaler()
scaler.fit(X_train)

#fit data
X_test_mmscalar = scaler.transform(X_test)
X_train_mmscalar = scaler.transform(X_train)


# In[11]:


#Instantiate the model
logit = LogisticRegression()
#fit the model
logit.fit(X_train, y_train)


# In[12]:


#export as pickle file for model deployment
pickle.dump(logit, open("model_to_deploy.pkl", "wb"))


# ###### The exported pickle model would be deployed using Heroku and a python app file.
