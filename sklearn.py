#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('git clone https://github.com/pksvv/25JRegression.git')


# In[ ]:





# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn

import seaborn as sns
sns.set()


# In[9]:


df = pd.read_csv('D:\\PYTHON WITH PRASHANT\\USA_Housing.csv')


# In[10]:


df.head()


# In[11]:


df.info()


# In[12]:


df.info()


# In[13]:


sns.pairplot(df)


# In[19]:


sns.distplot(df[ 'Avg. Area Income' ])


# In[20]:


plt.figure(figsize=(15,8))


# In[21]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
            
           


# In[22]:


# Implementing in sklearn
df.columns


# In[23]:


x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population',]]

y = df['Price']


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=42)


# In[29]:


df.shape


# In[30]:


x_train.shape


# In[32]:


x_test.shape


# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


lm = LinearRegression()
lm.fit(x_train,y_train)


# In[35]:


# Evaluation 


# In[37]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[38]:


y_pred = lm.predict(x_test)


# In[40]:


y_pred


# In[42]:


print (f'Mean Absolute Error : {mean_absolute_error(y_test,y_pred)}')
print (f'Mean Squared Error : {mean_squared_error(y_test,y_pred)}')
print (f'Root Mean Squared Error : {np.sqrt(mean_squared_error(y_test,y_pred))}')
print (f'R-Squared : {r2_score(y_test,y_pred)}')


# In[43]:


df.describe()


# In[ ]:




