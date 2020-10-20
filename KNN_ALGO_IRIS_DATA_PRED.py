#!/usr/bin/env python
# coding: utf-8

# # IRIS DATA KNN ALGORITH PREDICTION

# In[28]:



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[29]:


irs=sns.load_dataset("iris")


# In[30]:


irs.head()


# In[31]:


from sklearn.model_selection import train_test_split
X=irs.drop('species',axis=1)
y=irs["species"]


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[33]:


from sklearn.neighbors import KNeighborsClassifier


# In[34]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[35]:


knn.fit(X_train,y_train)


# In[36]:


pred=knn.predict(X_test)


# In[37]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[38]:


new=irs.copy()
pdd=new.drop(['species'],axis=1)


# In[39]:


pd.DataFrame(knn.predict(pdd),columns=irs.columns[-1:])


# In[40]:


#ALL THE BEST


# In[ ]:





# In[ ]:




