#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Wednesday Special : Data Exploration | #DS360withAkanksha|Week3


# In[1]:


# Importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Bringing in training data

#iris = pd.read_csv('iris.csv')

df_train = pd.read_csv('train_hp.csv')


# In[3]:


# Looking at columns name in the dataframe

df_train.columns


# In[4]:


# Target variable descriptive statistics

df_train['SalePrice'].describe()


# In[5]:


# Data distribution analysis using histogram

sns.distplot(df_train['SalePrice']);


# In[ ]:


#Observation:
'''
1) Deviate from Normal Distribution
2) Positive skewness
3) Show peakedness

'''


# In[6]:


# Statistical Analysis : Skewness and Kurtosis

print("Skewness: %f" %df_train['SalePrice'].skew())
print("Kurtosis: %f" %df_train['SalePrice'].kurt())


# In[7]:


# Relationship with numerical variable : Scatterplot lotarea/saleprice

var = 'LotArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));


# In[8]:


# Relationship with numerical variable : scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[9]:


# Relationship with numerical variable :scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[11]:


# Relationship with categorical features : boxplot overallqual/saleprice

var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=900000);


# In[14]:


var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=900000);
plt.xticks(rotation=90);


# In[17]:


# Correlation Matrix of all the variables in the training dataset

cor_matrix = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(cor_matrix, vmax=.8, square=True);


# In[18]:


# Zoomed heatmap style : saleprice correlation matrix

k = 8 #number of variables for heatmap
cols = cor_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[19]:


# Seaborn scatterplot

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# In[20]:


# Exploring missing data

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[25]:


#Standardizing data : Univariate Analysis
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[26]:


# Data Exploration : Bivariate analysis saleprice/grlivarea

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,900000));


# In[29]:


# Data Exploration : Bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,900000));


# In[30]:


# Indepth statistical analysis : Histogram and Normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[31]:


# Treating  positive skewness through log transformations

df_train['SalePrice'] = np.log(df_train['SalePrice'])


# In[32]:


# Histogram and Normal Probability plot after log transformation

sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[33]:


# Indepth statistical analysis : Histogram and Normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# In[34]:


# Treating skewness through log transformations

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])


# In[35]:


# Histogram and Normal Probability plot after log transformation

sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# In[36]:


# Indepth statistical analysis : Histogram and Normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# In[37]:


# Creating new variable since the variable TotalBsmtSF... 
#...has zero values which could affect log transformation


# Applying condition - if area>0 assign 1, for area===0 assign 0

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.assign[df_train['TotalBsmtSF']>0, 'HasBsmt'] = 1



# In[42]:


# Treating skewness through log transformations

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])


# In[43]:


# Indepth statistical analysis : Histogram and Normal probability plot


sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# In[44]:


# Checking homoscedasticity i.e.variables randomness | SalePrice vs GrLivArea

'''
Assuming a variable is homoscedastic when in reality it is heteroscedastic 
/ˌhɛtəroʊskəˈdæstɪk/) results in unbiased but inefficient point estimates
and in biased estimates of standard errors, and may result in overestimating 
the goodness of fit as measured by the Pearson coefficient.
'''

plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);


# In[45]:


# Checking homoscedasticity i.e.variables randomness | SalePrice vs TotalBsmtSF

plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);


# In[46]:


# Convering categorical variable into dummy
df_train = pd.get_dummies(df_train)


# In[ ]:




