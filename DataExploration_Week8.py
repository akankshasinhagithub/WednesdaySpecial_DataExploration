#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DS360withAkanksha | Wednesday Special:Data Exploration | Week 8 - Exploratory Data Analysis of Amazon stock Data


# In[8]:


#Importing required packages

import pandas as pd
import numpy as np
from scipy.stats import norm

#Packages for plotting

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Uploading dataset for inspection and analysis

df = pd.read_csv('AMZN.csv')


# In[4]:


#Inspecting top rows

df.head()


# In[5]:


#Inspecting the datatype

df.info()


# Observation : In the above output, it is evident that the date's datatype is object. 
#     For futher analysis the data type should be changed to datetime datatype

# In[7]:


#Changing the date's datatype to datetime 

df['Date'] = pd.to_datetime(df['Date'])


# In[13]:


# Plotting the AMZN stock data

plt.figure(figsize=(15,5))
plt.plot('Date','Close',data=df)

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=0)


# Observation : In the above plot it is evident that the Amazon stock price took a hit 
#     between March 2020 and April 2020 as the COVID19 was on rise on USA and the entire 
#     supply chain was affected though post April 2020 Amazon recovered from March 2020 hit 
#     by applying strong business strategies and coming up with solutions to provide best 
#     customer services via Prime Now.

# In[16]:


# Creating new column 'Daily Lag'

df['Daily Lag'] = df['Close'].shift(1)
df.head()


# In[17]:


# Creating new column 'Daily Returns' to analyze the returns for any particular day

df['Daily Returns'] = (df['Daily Lag']/df['Close']) -1
df.head()


# In[18]:


# Plotting the 'Daily Returns'

df['Daily Returns'].hist()


# In[19]:


# Finding descriptive statistics of 'Daily Returns'

mean = df['Daily Returns'].mean()
std = df['Daily Returns'].std()
print('mean =',mean)
print('Std deviation =',std)


# In[22]:


# Changing the histogram bin size to 30

df['Daily Returns'].hist(bins=30)
plt.axvline(mean,color='red',linestyle='dashed',linewidth=3)

plt.axvline(std,color='g',linestyle='dashed',linewidth=2)
plt.axvline(-std,color='g',linestyle='dashed',linewidth=2)


# In[23]:


# Checking Kurtosis

#Note: Kurtosis let us know about the presence of extreme values
# A/C to Wiki : "In probability theory and statistics, kurtosis is a measure of... 
#...the "tailedness" of the probability distribution of a real-valued random variable"

df['Daily Returns'].kurtosis()


# Observation: The skewness for this 'Daily Returns' is 1.86.  A positive skewness indicates that the 
#     size of the right-handed tail is larger than the left-handed tail. Additionally,  the value is positive, so the possibility of having extreme values is very less.

# In[ ]:




