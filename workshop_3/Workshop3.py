
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Load dataset for linear regression
df_energy = pd.read_csv('energydata.csv')
df_energy


# In[3]:


# Remove date column, rv1, and rv2 columns
df_energy.drop('date', axis=1, inplace=True) # axis=1 to indicate we're dropping a column
df_energy.drop('rv1', axis=1, inplace=True)
df_energy.drop('rv2', axis=1, inplace=True)
df_energy


# In[4]:


# Sum appliances and lights for total energy
df_energy['energy'] = df_energy['Appliances'] + df_energy['lights']
df_energy


# In[5]:


# Then drop appliances and lights as they provide too much information
df_energy.drop('Appliances', axis=1, inplace=True)
df_energy.drop('lights', axis=1, inplace=True)
df_energy


# In[6]:


df_energy_X = df_energy.drop('energy', axis=1)
df_energy_y = df_energy['energy']
df_energy_X


# In[7]:


from sklearn.linear_model import LinearRegression as LR
def PF(X, degree):
    X_np = pd.DataFrame.as_matrix(X)
    tmp_raise = X_np
    for i in range(2, degree+1):
        tmp_raise = np.append(tmp_raise, np.power(X_np, i), axis=1)
    X_np = tmp_raise
    finalarr = np.ones((X_np.shape[0], X_np.shape[1]+1))
    finalarr[:,:-1] = X_np
    return finalarr






# In[8]:


# Do 60-20-20 train_test_split for training set, validation set, and test set
from sklearn.model_selection import train_test_split

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)

    #print('X_train percentage: {}, X_val percentage: {}, X_test percentage: {}'
    #     .format(len(X_train)/len(X), 
    #             len(X_val)/len(X), 
    #             len(X_test)/len(X)))
    return X_train, X_val, X_test, y_train, y_val, y_test


# In[9]:


validation_acc = []
for i in range(1,15):
    X_poly = PF(df_energy_X, i)
    X_train, X_val, X_test, y_train, y_val, y_test = split(X_poly, df_energy_y)
    LR_model = LR()
    LR_model.fit(X_train, y_train)
    print('training set R^2 for degree {}: {}'.format(i, 
                                                       LR_model.score(X_train, y_train)))
    print('validation set R^2 for degree {}: {}'. format(i,
                                                         LR_model.score(X_val, y_val)))
    validation_acc.append(LR_model.score(X_val, y_val))
validation_acc


# In[10]:


import matplotlib.pyplot as plt
plt.plot([i+1 for i in range(len(validation_acc))], validation_acc)
plt.axis([1, 15, -0.1, 0.2])
plt.show()
    


# In[11]:


np_mnist = np.loadtxt(open("mnist_train.csv", "rb"), delimiter=",", skiprows=1)
np_mnist


# In[12]:


df_mnist = pd.DataFrame(np_mnist)
df_mnist


# In[13]:


df_mnist.rename(columns={0: 'label'}, inplace=True)
df_mnist


# In[14]:


df_mnist_X = df_mnist.drop('label', axis=1)
df_mnist_y = df_mnist['label']


# In[15]:


X_train, X_val, X_test, y_train, y_val, y_test = split(df_mnist_X, df_mnist_y)


# In[16]:


from sklearn.linear_model import LogisticRegression as LR

LRmodel = LR(solver='lbfgs')
LRmodel.fit(X_train, y_train)
print('Logistic Regression Training Accuracy: {}'.format(LRmodel.score(X_train, y_train)))
print('Logistic Regression Validation Accuracy: {}'.format(LRmodel.score(X_val, y_val)))


# In[17]:


# Cross validation code that we won't run now, but you can try it in your own time
# This should take approximately 10 minutes to run (go grab dinner in between?)
# Recalling that C is inversely proportional to regularization strength

'''
for i in range(0.3, 1.3, 0.1):
    LRmodel = LR(C=i, solver='lbfgs')
    LRmodel.fit(X_train, y_train)
    print('Logistic Regression Training Accuracy with C = {}: {}'
         .format(i, LRmodel.score(X_train, y_train)))
    print('Logistic Regression Validation Accuracy with C = {}: {}'
         .format(i, LRmodel.score(X_val, y_val)))
'''


    


# In[10]:


df_iris = pd.read_csv('iris.csv')
df_iris


# In[11]:


df_iris_X = df_iris.drop('species', axis=1)
df_iris_y = df_iris['species']
X_train, X_val, X_test, y_train, y_val, y_test = split(df_iris_X, df_iris_y)


# In[13]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf


# In[15]:


clf = clf.fit(X_train, y_train)
print('Decision Tree train set accuracy: {}'.format(clf.score(X_train, y_train)))
print('Decision Tree validation set accuracy: {}'.format(clf.score(X_val, y_val)))


# In[19]:


tree.export_graphviz(clf, out_file='tree.dot')    

