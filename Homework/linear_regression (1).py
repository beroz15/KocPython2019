#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np


# In[1]:


def linear_regression(covariates, targets):
    covariates = np.array(covariates)
    targets = np.array(targets).reshape(-1,1)
    combination = np.hstack([covariates, targets])
    covariates = covariates[~np.isnan(combination).any(axis=1)]
    targets = targets[~np.isnan(combination).any(axis=1)]
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(covariates), covariates)), np.transpose(covariates)), targets)
    y_hat = np.matmul(covariates, beta)
    errors = targets - y_hat
    for index, coeff in enumerate(beta.tolist()):
        mean = float(covariates[:, index].mean())
        summation = 0
        for point in covariates[:, index].tolist():
            summation += pow(float(point) - mean, 2)
        variance_beta.append(variance/summation)
    variance_beta = np.array(variance_beta).reshape(-1,1)
    se_beta = np.sqrt(variance_beta)
    lower_bounds = beta - (2 * se_beta)
    upper_bounds = beta + (2 * se_beta)
    return beta[:,0], se_beta[:,0], lower_bounds[:,0], upper_bounds[:,0]


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import pandas as pd


# In[6]:


import numpy as np


# In[7]:


pd.set_option('display.max_rows', 500)


# In[8]:


pd.set_option('display.max_columns', 500)


# In[9]:


pd.set_option('display.width', 1000)


# In[10]:


df = pd.read_csv("winequality-white.csv", sep=";")


# In[11]:


covariates = df.drop("quality", axis=1).values


# In[12]:


targets = df["quality"].values


# In[13]:


beta, se_beta, lower_bounds, upper_bounds = linear_regression(covariates, targets)


# In[14]:


result_table = pd.DataFrame.from_dict({"lower_bound_for_estimates": lower_bounds,
                                       "estimates": beta,
                                       "upper_bound_for_etimates": upper_bounds,
                                       "standard_errors": se_beta})


# In[15]:


beta, se_beta, lower_bounds, upper_bounds = linear_regression(covariates, targets)


# In[16]:


print("Result table:")


# In[17]:


display(result_table)


# In[18]:


plt.plot(lower_bounds)


# In[19]:


plt.plot(beta)


# In[20]:


plt.plot(upper_bounds)
plt.title("Result plot")
plt.legend(["lower_bound_for_estimates",
            "estimates",
            "upper_bound_for_estimates"])


# In[ ]:




