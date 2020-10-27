#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import sqrt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import csv
#import cvxopt

import Enet
import pre_pro


# In[2]:


#Import data from file
with open("new_superset_train.csv", newline='') as f:
#with open("training_data_new/Adelaide_train.csv", newline='') as f:
    reader = csv.reader(f)
    inputdata = list(reader)

data = []
for line in inputdata:
    data.append(list(map(float,line)))
     


# In[3]:


#Conduct Pre-processing
pp = pre_pro.pre_prosessing(data)
pp.zscore_remove(6)
pp.qrange_remove(0.05,0.95)
pp.x_y_split()
pp.closegame_remove(-7,7)
x_train, x_test, y_train, y_test = pp.data_split(test_size = 0.1, seed = 10)


# In[4]:


#Principal Component Analysis
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[5]:


enet= Enet.Eneter()
enet.fit(x_train, y_train)


# In[6]:


y_pred = enet.predict(x_test)

t_pos =0
f_pos = 0
t_neg = 0
f_neg = 0
for t,p in zip(y_test,y_pred):
    if(t >= 0 and p >= 0):
        t_pos+=1
    elif(t<0 and p >0):
        f_pos+=1
    elif(t<0 and p<0):
        t_neg+=1  
    elif(t>0 and p<0):
        f_neg+=1  
        
        
print("True Pos: ",t_pos)
print("False Pos: ",f_pos)
print("True neg: ",t_neg)
print("False neg: ",f_neg)

print("Accuracy: ", ((t_pos+t_neg)/(t_pos+f_pos+t_neg+f_neg)*100))

print('R2 = '+str(r2_score(y_test, y_pred, multioutput='variance_weighted')))
print('RMSE = '+str(sqrt(mean_squared_error(y_test, y_pred))))
print('RAE = '+str(mean_absolute_error(y_test, y_pred)))
print("True Range: ",min(y_test)," to ",max(y_test))
print("Pred Range: ",min(y_pred)," to ",max(y_pred))


# In[8]:


#GrandFinal Prediction
with open("grandF.csv", newline='') as f:
#with open("training_data_new/Adelaide_train.csv", newline='') as f:
    reader = csv.reader(f)
    inputdata = list(reader)

gfdata = []
for line in inputdata:
    gfdata.append(list(map(float,line)))

X = [item[:-1] for item in gfdata]
Y = [item[0] for item in gfdata]

gf = scaler.transform(X)

Prediction = enet.predict(gf)
print("Geelong vs Richmond:",Prediction)


# In[ ]:




