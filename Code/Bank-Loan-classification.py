#!/usr/bin/env python
# coding: utf-8

# # Bank Loan Term Prediction
# ---

# ## Import packages & read data.

# In[1]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# Visualization imports
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# %config InlineBackend.figure_format = 'svg'

# Modeling imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score,log_loss, confusion_matrix, precision_score, recall_score, accuracy_score 
from sklearn import linear_model, ensemble , tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , VotingClassifier
from sklearn.linear_model import LogisticRegression
import imblearn.over_sampling
from sklearn.svm import SVC  
from sklearn.utils import class_weight
import statsmodels.api as sm
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline, make_pipeline 
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV
import sklearn.metrics as metrics


# In[2]:


df = pd.read_csv('credit_train.csv')
df


# In[3]:


df.shape


# In[4]:


df.tail()


# In[5]:


df.isna().sum()


# In[6]:


df.info() 


# In[7]:


duplicate = df.duplicated()
print(f'Duplicate in df :', duplicate.sum())


# **Rename columns for easer code writing**

# In[8]:


df.columns


# In[9]:


df.columns = df.columns.str.replace(' ','_')


# In[10]:


df.columns


# ## Split the data for train, validation and test

# In[11]:


# X = df.drop(columns='Term')
# y = pd.DataFrame(df['Term'])

# cross val
df_train, df_test  = train_test_split(df, test_size=0.2, random_state=42)

# # split the data for train and test
# df_Train, df_test = train_test_split(df, test_size = 0.2, random_state = 30 )

# # split the train for train and val
# df_train, df_val = train_test_split(df_Train, test_size = 0.2, random_state = 30 )


# In[12]:


print(f'Shape of train:', df_train.shape)
# print(f'Shape of validation:', df_val.shape)
print(f'Shape of test:', df_test.shape)


# ## Data Pre-processing

# ### Cleaning data

# In[13]:


# reset index for train
df_train = df_train.reset_index(drop=True)

# # reset index for val
# df_val = df_val.reset_index(drop=True)

# reset index for val
df_test = df_test.reset_index(drop=True)


# In[14]:


# dope nulls in Loan_ID

# for train
df_train = df_train.dropna(subset = ['Loan_ID'])

# # for val
# df_val = df_val.dropna(subset = ['Loan_ID'])

# for test
df_test = df_test.dropna(subset = ['Loan_ID'])


# In[15]:


print(f'Shape of train:', df_train.shape)
# print(f'Shape of validation:', df_val.shape)
print(f'Shape of test:', df_test.shape)


# In[16]:


df_train.sample(20)


# In[17]:


df_train.info()


# In[18]:


df_train.isna().sum()


# In[19]:


# check for dublicate

# for train
duplicate = df_train.duplicated()
print(f'Duplicate in train :', duplicate.sum())

# # for val
# duplicate = df_val.duplicated()
# print(f'Duplicate in validation :', duplicate.sum())

# for test
duplicate = df_test.duplicated()
print(f'Duplicate in test :', duplicate.sum())
print(f'Shape of train:', df_train.shape)
# print(f'Shape of validation:', df_val.shape)
print(f'Shape of test:', df_test.shape)


# In[20]:


# drop duplicates rows
# train
df_train.drop_duplicates(inplace=True)

# val
# df_val.drop_duplicates(inplace=True)

# test
df_test.drop_duplicates(inplace=True)


# In[21]:


# check for dublicate

# for train
duplicate = df_train.duplicated()
print(f'Duplicate in train :', duplicate.sum())

# # for val
# duplicate = df_val.duplicated()
# print(f'Duplicate in validation :', duplicate.sum())

# for test
duplicate = df_test.duplicated()
print(f'Duplicate in test :', duplicate.sum())
print(f'Shape of train:', df_train.shape)
# print(f'Shape of validation:', df_val.shape)
print(f'Shape of test:', df_test.shape)


# **Duplicate in Loan ID**

# In[22]:


df_train['Loan_ID'].value_counts().sort_values(ascending=False)


# In[23]:


df_train[df_train.Loan_ID.duplicated()]


# In[24]:


df_train[df_train['Loan_ID'] == '7830a00a-20c4-4480-9cf0-fe2f86b5266b']


# In[25]:


df_train[df_train['Loan_ID'] == '5a90cbe3-8fee-4582-8823-1f31546dec6e']


# We can see an error in data entry. There is a duplicate in loan ID but the difference in current loan amount or null values,
# 
# **Now we fix it.**

# In[26]:


df_train[(df_train.Loan_ID.duplicated() & (df_train['Current_Loan_Amount'] == 99999999.0))]


# In[27]:


# drop duplicate in Loan_ID and Current_Loan_Amount = 99999999.0

# for train
df_train = df_train[~(df_train.Loan_ID.duplicated() & (df_train['Current_Loan_Amount'] == 99999999.0))]

# for val
# df_val = df_val[~(df_val.Loan_ID.duplicated() & (df_val['Current_Loan_Amount'] == 99999999.0))]

# for test
df_test = df_test[~(df_test.Loan_ID.duplicated() & (df_test['Current_Loan_Amount'] == 99999999.0))]


# In[28]:


df_train[(df_train.Loan_ID.duplicated())]


# In[29]:


df_train[df_train['Loan_ID'] == 'ff486b10-f97d-4dff-bb98-436ef48d8ab1']


# In[30]:


# dope nulls in Loan_Status

# for train
df_train = df_train.dropna(subset = ['Annual_Income'])

# # for val
# df_val = df_val.dropna(subset = ['Annual_Income'])

# for test
df_test = df_test.dropna(subset = ['Annual_Income'])


# In[31]:


df_train[df_train['Loan_ID'] == 'ff486b10-f97d-4dff-bb98-436ef48d8ab1']


# In[32]:


#df_train
df_train.Purpose.unique()
# #df_val
# df_val.Purpose.unique()
#da_test
df_test.Purpose.unique()


# In[33]:


df_train.Purpose.value_counts()


# In[34]:


#df_train
df_train.Purpose = df_train.Purpose.str.replace('other','Other')
# #df_val
# df_val.Purpose = df_val.Purpose.str.replace('other','Other')
#df_test
df_test.Purpose = df_test.Purpose.str.replace('other','Other')


# In[35]:


df_train.Purpose.value_counts()


# In[36]:


df_train.Purpose.unique()


# In[37]:


df_train.isnull().sum() # train


# In[38]:


# dope duplicated in Loan_ID

# for train
df_train = df_train.drop_duplicates(subset = ['Loan_ID'])

# # for val
# df_val = df_val.drop_duplicates(subset = ['Loan_ID'])

# for test
df_test = df_test.drop_duplicates(subset = ['Loan_ID'])


# In[39]:


df_train.isnull().sum() # train


# In[40]:


print(f'Shape of train:', df_train.shape)
# print(f'Shape of validation:', df_val.shape)
print(f'Shape of test:', df_test.shape)


# In[41]:


plt.figure(figsize=(10,5))
sns.countplot(df_train['Years_in_current_job'], palette='pink_r');


# In[42]:


# fill nulls in Years_in_current_job 

# for train
df_train['Years_in_current_job'] = df_train['Years_in_current_job'].fillna('10+ years')

# # for val
# df_val['Years_in_current_job'] = df_val['Years_in_current_job'].fillna('10+ years')

# for test
df_test['Years_in_current_job'] = df_test['Years_in_current_job'].fillna('10+ years')


# In[43]:


# drop Months_since_last_delinquent bc the null > 50&

# train
df_train = df_train.drop(columns='Months_since_last_delinquent')

# test
df_test = df_test.drop(columns='Months_since_last_delinquent')


# In[44]:


df_train.isnull().sum()


# In[45]:


# drop nulls 

# for train
df_train = df_train.dropna()

# # for val
# df_val = df_val.dropna()

# for test
df_test = df_test.dropna()


# In[46]:


df_train.isnull().sum()


# In[47]:


df_train.duplicated().sum()


# ## Feature Engneering
# ---

# In[48]:


df_train.isnull().sum()


# In[49]:


df_train.info()


# ### Get Dummies

# In[50]:


# train
bank_lone_train = pd.get_dummies(df_train, columns =['Term','Home_Ownership','Purpose','Loan_Status', 'Years_in_current_job'], drop_first=True) ###

# # val
# bank_lone_val = pd.get_dummies(df_val, columns =['Term','Home_Ownership','Purpose','Loan_Status', 'Years_in_current_job'], drop_first=True) ###

# test
bank_lone_test = pd.get_dummies(df_test, columns =['Term','Home_Ownership','Purpose','Loan_Status' , 'Years_in_current_job'], drop_first=True) ###


# In[51]:


bank_lone_train.columns


# In[52]:


df_train.corr()


# In[53]:


plt.figure(figsize=(10,8))

# corr
data_corr = df_train.corr()
# data_corr = bank_lone_train.corr()

# mask
mask = np.triu(np.ones_like(data_corr, dtype=np.bool))

# adjust mask and df
mask = mask[1:, :-1]
corr = data_corr.iloc[1:,:-1].copy()

sns.heatmap(corr, cmap = 'pink_r', annot = True, vmin= -1, vmax= 1, linewidths=1.5, fmt='.2f', mask=mask);
plt.title('CORRELATION BETWEEN FEATURES\n', loc='left', fontsize=18);
# plt.savefig('plot13.png', dpi = 300, bbox_inches = 'tight');


# In[54]:


# sns.pairplot(bank_lone_train, hue = 'Term_Short Term', palette = 'pink_r');


# ## Visualize data
# ___

# In[55]:


c = ['#724949','#cfa691', '#120f0f', '#a06868']
plt.figure(figsize=(7,7))
plt.pie(x = bank_lone_train['Term_Short Term'].value_counts(),
        labels=['Short term','Long term'],autopct='%.2f%%',
        textprops={'fontsize': 12},explode=[0,0.09], colors = ['#724949','#DEDCBB'])
plt.title('Time Period of Taking Loan',fontdict={'fontsize':15});


# In[56]:


plt.figure(figsize=(10,9))
sns.countplot(y='Purpose' , data=df_train, order = df_train['Purpose'].value_counts().index,
              hue='Term', palette = 'pink_r')
plt.title('Purpose of taking Loan' , fontdict={'fontsize':20})
plt.legend(title="Loan type", loc="lower right");


# In[57]:


plt.figure(figsize=(10,8))
sns.countplot(x='Home_Ownership',data=df_train ,order = df_train['Home_Ownership'].value_counts().index
              ,hue='Term',  palette = 'pink_r')
plt.title('Own Property vs Loan Status',fontdict={'fontsize':20})
plt.legend(title="Loan type", loc="upper right", labels=["Short Term","Long Term"]);


# ### Droping outliers

# In[58]:


plt.figure(figsize = [15,20])
plt.subplot(3,2,1)
sns.boxplot(x='Term_Short Term',y='Current_Loan_Amount',
            palette='pink_r', data=bank_lone_train.sort_values('Current_Loan_Amount',ascending=False));
plt.title('Before dropping outliers',fontsize = 15 )

bank_lone_train = bank_lone_train[bank_lone_train['Current_Loan_Amount'] != 99999999]
bank_lone_train = bank_lone_train[((bank_lone_train['Current_Loan_Amount'] <= 600000 )
                                   & (bank_lone_train['Term_Short Term']==1))
                                  | (bank_lone_train['Term_Short Term']==0)]

plt.subplot(3,2,2)
sns.boxplot(x='Term_Short Term',y='Current_Loan_Amount',
            palette='pink_r', data=bank_lone_train.sort_values('Current_Loan_Amount',ascending=False));
plt.title('After dropping outliers',fontsize = 15 );


# In[59]:


bank_lone_test = bank_lone_test[bank_lone_test['Current_Loan_Amount'] != 99999999]
bank_lone_test = bank_lone_test[((bank_lone_test['Current_Loan_Amount'] <= 600000 )
                                   & (bank_lone_test['Term_Short Term']==1))
                                  | (bank_lone_test['Term_Short Term']==0)]


# In[60]:


plt.figure(figsize = [15,20])
plt.subplot(3,2,1)
sns.boxplot(x='Term_Short Term',y='Credit_Score',
            palette='pink_r', data = bank_lone_train.sort_values('Credit_Score',ascending=False));
plt.title('Before dropping outliers',fontsize = 15 )

bank_lone_train = bank_lone_train.loc[bank_lone_train['Credit_Score'] <= 1500,:]
bank_lone_train = bank_lone_train.loc[bank_lone_train['Credit_Score'] >= 620 ,:]
bank_lone_train = bank_lone_train[((bank_lone_train['Credit_Score'] >= 680 )
                                   & (bank_lone_train['Term_Short Term']==1))| 
                                  (bank_lone_train['Term_Short Term']==0)]

plt.subplot(3,2,2)
sns.boxplot(x='Term_Short Term',y='Credit_Score',
            palette='pink_r', data = bank_lone_train.sort_values('Credit_Score',ascending=False));
plt.title('After dropping outliers',fontsize = 15 );


# In[61]:


bank_lone_test = bank_lone_test.loc[bank_lone_test['Credit_Score'] <= 1500,:]
bank_lone_test = bank_lone_test.loc[bank_lone_test['Credit_Score'] >= 620 ,:]
bank_lone_test = bank_lone_test[((bank_lone_test['Credit_Score'] >= 680 )
                                   & (bank_lone_test['Term_Short Term']==1))| 
                                  (bank_lone_test['Term_Short Term']==0)]


# In[62]:


plt.figure(figsize = [15,20])
plt.subplot(3,2,1)
sns.boxplot(x='Term_Short Term',y='Annual_Income',
            palette='pink_r', data = bank_lone_train.sort_values('Annual_Income',ascending=False));
plt.title('Before dropping outliers',fontsize = 15 )

bank_lone_train = bank_lone_train.loc[bank_lone_train['Annual_Income'] <= 2750000,:]
bank_lone_train = bank_lone_train[((bank_lone_train['Annual_Income'] <= 2395000 )
                                   & (bank_lone_train['Term_Short Term']==1))
                                  | (bank_lone_train['Term_Short Term']==0)]

plt.subplot(3,2,2)
sns.boxplot(x='Term_Short Term',y='Annual_Income',
            palette='pink_r', data = bank_lone_train.sort_values('Annual_Income',ascending=False));
plt.title('After dropping outliers',fontsize = 15 );


# In[63]:


bank_lone_test = bank_lone_test.loc[bank_lone_test['Annual_Income'] <= 2750000,:]
bank_lone_test = bank_lone_test[((bank_lone_test['Annual_Income'] <= 2395000 )
                                   & (bank_lone_test['Term_Short Term']==1))
                                  | (bank_lone_test['Term_Short Term']==0)]


# In[64]:


plt.figure(figsize = [15,20])
plt.subplot(3,2,1)
sns.boxplot(x='Term_Short Term',y='Monthly_Debt',
            palette='pink_r', data=bank_lone_train.sort_values('Monthly_Debt',ascending=False));
plt.title('Before dropping outliers',fontsize = 15)

bank_lone_train = bank_lone_train.loc[bank_lone_train['Monthly_Debt'] <= 44500,:]
bank_lone_train = bank_lone_train[((bank_lone_train['Monthly_Debt'] <= 36000 )& (bank_lone_train['Term_Short Term']==1))| 
                                  (bank_lone_train['Term_Short Term']==0)]

plt.subplot(3,2,2)
sns.boxplot(x='Term_Short Term',y='Monthly_Debt',
            palette='pink_r', data=bank_lone_train.sort_values('Monthly_Debt',ascending=False));
plt.title('After dropping outliers',fontsize = 15 );


# In[65]:


bank_lone_test = bank_lone_test.loc[bank_lone_test['Monthly_Debt'] <= 44500,:]
bank_lone_test = bank_lone_test[((bank_lone_test['Monthly_Debt'] <= 36000 )& 
                                   (bank_lone_test['Term_Short Term']==1))| 
                                  (bank_lone_test['Term_Short Term']==0)]


# In[66]:


plt.figure(figsize = [15,20])
plt.subplot(3,2,1)
sns.boxplot(x='Term_Short Term',y='Current_Credit_Balance',
            palette='pink_r', data=bank_lone_train.sort_values('Current_Credit_Balance',ascending=False));
plt.title('Before dropping outliers',fontsize = 15 )

bank_lone_train = bank_lone_train.loc[bank_lone_train['Current_Credit_Balance'] <= 760000,:]
bank_lone_train = bank_lone_train[((bank_lone_train['Current_Credit_Balance'] <= 504000 )& 
                                   (bank_lone_train['Term_Short Term']==1))| (bank_lone_train['Term_Short Term']==0)]

plt.subplot(3,2,2)
sns.boxplot(x='Term_Short Term',y='Current_Credit_Balance',
            palette='pink_r', data=bank_lone_train.sort_values('Current_Credit_Balance',ascending=False));
plt.title('After dropping outliers',fontsize = 15 );


# In[67]:


bank_lone_test = bank_lone_test.loc[bank_lone_test['Current_Credit_Balance'] <= 760000,:]
bank_lone_test = bank_lone_test[((bank_lone_test['Current_Credit_Balance'] <= 504000 )& 
                                   (bank_lone_test['Term_Short Term']==1))| 
                                  (bank_lone_test['Term_Short Term']==0)]


# In[68]:


plt.figure(figsize = [15,20])
plt.subplot(3,2,1)
sns.boxplot(x='Term_Short Term',y='Maximum_Open_Credit',
            palette='pink_r', data=bank_lone_train.sort_values('Maximum_Open_Credit',ascending=False));
plt.title('Before dropping outliers',fontsize = 15)

bank_lone_train = bank_lone_train.loc[bank_lone_train['Maximum_Open_Credit'] <= 1400000,:]
bank_lone_train = bank_lone_train[((bank_lone_train['Maximum_Open_Credit'] <= 990000 )& 
                                   (bank_lone_train['Term_Short Term']==1))| (bank_lone_train['Term_Short Term']==0)]
plt.subplot(3,2,2)
sns.boxplot(x='Term_Short Term',y='Maximum_Open_Credit',
            palette='pink_r', data=bank_lone_train.sort_values('Maximum_Open_Credit',ascending=False));
plt.title('After dropping outliers',fontsize = 15 );


# In[69]:


bank_lone_test = bank_lone_test.loc[bank_lone_test['Maximum_Open_Credit'] <= 1400000,:]
bank_lone_test = bank_lone_test[((bank_lone_test['Maximum_Open_Credit'] <= 990000 )& 
                                   (bank_lone_test['Term_Short Term']==1))| 
                                  (bank_lone_test['Term_Short Term']==0)]


# ### plot the correlation after one hot coding

# In[70]:


plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(bank_lone_train.corr()[
    ['Term_Short Term']].sort_values(by='Term_Short Term',ascending=False),
                      vmin=-1, vmax=1, annot=True,
                      cmap = 'pink_r')
plt.title('CORRELATION BETWEEN FEATURES AND TARGET AFTER ONE HOT CODING\n', loc='center', fontsize=18);


# In[71]:


X_train = bank_lone_train.drop(['Term_Short Term','Loan_ID','Customer_ID',
                                'Credit_Score', 'Years_of_Credit_History', 
                                'Number_of_Credit_Problems', 'Number_of_Open_Accounts',
                                'Bankruptcies'], axis = 1)
y_train = bank_lone_train['Term_Short Term']
X_test = bank_lone_test.drop(['Term_Short Term','Loan_ID','Customer_ID',
                                'Credit_Score', 'Years_of_Credit_History', 
                                'Number_of_Credit_Problems', 'Number_of_Open_Accounts',
                                'Bankruptcies'], axis = 1)
y_test = bank_lone_test['Term_Short Term']


# In[72]:


model = sm.OLS(y_train,X_train)
fit = model.fit()
fit.summary()


# ---

# In[73]:


y_train.value_counts()


# In[74]:


# Separate class
long_term_0 = bank_lone_train[bank_lone_train['Term_Short Term'] == 0]
short_term_1 = bank_lone_train[bank_lone_train['Term_Short Term'] == 1]# print the shape of the class
print('Long term 0:', long_term_0.shape[0])
print('Short term 1:', short_term_1.shape[0])


# ## Logistic Regression
# ---

# In[75]:


# gridsearch
params = {'C': [0.001,0.01,0.1,1,10,100,1000], 'class_weight':[{0:0,0:0.01,0:0.1,0:0.5,0:1,0:10,0:2} ] }
lr_grid = GridSearchCV(LogisticRegression(), param_grid = params, scoring='f1', cv = 5)
lr_grid.fit(X_train, y_train)
print('\n Best param after grid search: ', lr_grid.best_params_ )
print(' Best f1_score for cross validation: ',lr_grid.best_score_ )

# normal
LR = LogisticRegression(C= 0.001 ,solver='liblinear')
kf = KFold(n_splits=10, random_state=42, shuffle=True)
cr_f1 = cross_val_score(LR, X_train, y_train, scoring='f1', cv=kf)
print('\n Normal Logistic Regression Valdition F1: \n',cr_f1)
print('\n Mean Normal Logistic Regression Valdition F1: \n',cr_f1.mean())
print('--------------------------------')

# balenced
lr_balanced = LogisticRegression(class_weight='balanced', solver='liblinear')
cr_balnced_f1 = cross_val_score(lr_balanced, X_train, y_train, scoring='f1', cv=kf)
print('\n Balanced class weights Logistic Regression Valdition F1: \n',cr_balnced_f1)
print('\n Mean Balanced class weights Logistic Regression Valdition F1: \n',cr_balnced_f1.mean())
print('--------------------------------')

# weighted
lr_4x = LogisticRegression(C= 0.001, class_weight={0 : 2, 1 : 1}, solver='liblinear')
cr_weghts_f1 = cross_val_score(lr_4x, X_train, y_train, scoring='f1', cv=kf)
print('\n Class weights Logistic Regression Valdition F1: \n',cr_weghts_f1)
print('\n Mean Class weights Logistic Regression Valdition F1: \n',cr_weghts_f1.mean())
print('--------------------------------')

# smote
imba_pipeline = make_pipeline(SMOTE(random_state=42), 
                              LogisticRegression(C= 0.001, solver='liblinear'))

imba_val = cross_val_score(imba_pipeline, X_train, y_train, scoring='f1', cv=kf)
print('\n Smote Logistic Regression Valdition F1: \n', imba_val)
print('\n Mean Smote Logistic Regression Valdition F1: \n', imba_val.mean())


# In[76]:


LR.fit(X_train, y_train)
precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_train, LR.predict_proba(X_train)[:,1] )
plt.plot(threshold_curve, precision_curve[1:],label='precision', color = '#724949')
plt.plot(threshold_curve, recall_curve[1:], label='recall', color = '#DEDCBB')
plt.legend(loc='lower left')
plt.xlabel('Threshold (above this probability)');
plt.title('Precision and Recall Curves');


# In[77]:


y_predict = (LR.predict_proba(X_train)[:, 1] >= 0.65)

print("Default threshold:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_train, y_predict), 
                                                     recall_score(y_train, y_predict)))


# In[78]:


y_predict = (LR.predict_proba(X_train)[:, 1] >= 0.624)

loan_confusion = confusion_matrix(y_train, y_predict)

sns.heatmap(loan_confusion , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Train Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# In[79]:


# confusion matrix for crossval
y_pred = cross_val_predict(LR, X = X_train, y = y_train, cv=5)
conf_mat = confusion_matrix(y_train, y_pred)
sns.heatmap(conf_mat , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Cross Validation Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# ## KNN Model
# ---

# In[80]:


knn = KNeighborsClassifier(n_neighbors= 9)
k_range = list(range(3,11))
param_grd = dict(n_neighbors=k_range)
grid = GridSearchCV(KNeighborsClassifier(), param_grd, scoring='f1', cv = 5)
grid.fit(X_train, y_train)
print('Best estimator: ', grid.best_estimator_ )
print('Best f1_score for cross validation: ',grid.best_score_ )


# In[81]:


y_predict = (grid.predict_proba(X_train)[:, 1] >= 0.624)

loan_confusion = confusion_matrix(y_train, y_predict)

sns.heatmap(loan_confusion , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Train Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# ## Decision Tree Classifier
# ---

# In[82]:


# normal
Decision_Tree = DecisionTreeClassifier(max_depth = 8)
Decision_Tree.fit(X_train, y_train)
scores = cross_val_score(Decision_Tree, X_train, y_train, cv=5, scoring='f1')
print('Normal Decision Tree Valdition F1:',scores.mean())


# balenced
dt_bal = DecisionTreeClassifier(max_depth = 8, class_weight='balanced')
dt_bal.fit(X_train, y_train)
scores = cross_val_score(dt_bal, X_train, y_train, cv=5, scoring='f1')
print('Balanced class weights Decision Tree Valdition F1:',scores.mean())


# weighted
dt_wtd = DecisionTreeClassifier(class_weight= {0 : 10, 1 : 1})
scores = cross_val_score(dt_wtd, X_train, y_train, cv=5, scoring='f1')
dt_wtd.fit(X_train, y_train)
print('10:1 class weights Decision Tree Valdition F1:',scores.mean())

#gridsearch

tree_param = {'criterion':['gini','entropy'],
              'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}

gd_sr = GridSearchCV(DecisionTreeClassifier(), param_grid=tree_param,scoring='f1',cv=5,n_jobs=-1)
gd_sr.fit(X_train, y_train)
best_parameters = gd_sr.best_params_
print('\n Best param after grid search', best_parameters)
print('\n Best score after grid search', gd_sr.best_score_)


# In[84]:


precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_train, Decision_Tree.predict_proba(X_train)[:,1] )
plt.plot(threshold_curve, precision_curve[1:],label='precision', color = '#724949')
plt.plot(threshold_curve, recall_curve[1:], label='recall', color = '#DEDCBB')
plt.legend(loc='lower left')
plt.xlabel('Threshold (above this probability)');
plt.title('Precision and Recall Curves');


# In[85]:


y_predict = (Decision_Tree.predict_proba(X_train)[:, 1] >= 0.5569)

print("Default threshold:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_train, y_predict), 
                                                     recall_score(y_train, y_predict)))


# In[88]:


y_predict = (Decision_Tree.predict_proba(X_train)[:, 1] >= 0.61)

loan_confusion = confusion_matrix(y_train, y_predict)

sns.heatmap(loan_confusion , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Train Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# In[89]:


# confusion matrix for crossval
y_pred = cross_val_predict(Decision_Tree, X = X_train, y = y_train, cv=5)
conf_mat = confusion_matrix(y_train, y_pred)
sns.heatmap(conf_mat , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Cross Validation Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# ## Random Forest Classifier
# ---

# In[91]:


# normal
Random_Forest = RandomForestClassifier(n_estimators = 5, random_state=1)
Random_Forest.fit(X_train, y_train)
scores = cross_val_score(Random_Forest, X_train, y_train, cv=10, scoring='f1')
print('\n Normal Random Forest Valdition F1: \n',scores)
print('\n Mean Normal Random Forest Valdition F1:  \n',scores.mean())

# balenced
rf_bal = RandomForestClassifier(n_estimators = 10, random_state=1, class_weight='balanced')
rf_bal.fit(X_train, y_train)
scores = cross_val_score(rf_bal, X_train, y_train, cv=10, scoring='f1')
print('\n Balanced class weights Random Forest Valdition F1: \n',scores)
print('\n Mean Balanced class weights Random Forest Valdition F1: \n',scores.mean())

# weighted
rf_wtd = RandomForestClassifier(n_estimators = 10, random_state=1, class_weight= {0 : 2, 1 : 1})
rf_wtd.fit(X_train, y_train)
scores = cross_val_score(rf_wtd, X_train, y_train, cv=10, scoring='f1')
print('\n 2:1 class weights Random Forest Valdition F1:\n',scores)
print('\n 2:1 class weights Random Forest Valdition F1: \n',scores.mean())

#gridsearch
grid_param = {
    'n_estimators': [100, 300, 200, 50, 500],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]}
gd_sr1 = GridSearchCV(estimator=Random_Forest, param_grid=grid_param,scoring='f1',cv=5,n_jobs=-1)
gd_sr1.fit(X_train, y_train)
best_parameters = gd_sr1.best_params_
print('\n Best param after grid search', best_parameters)
print('\n Best score after grid search', gd_sr1.best_score_)


# In[92]:


precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_train, Random_Forest.predict_proba(X_train)[:,1] )
plt.plot(threshold_curve, precision_curve[1:],label='precision', color = '#724949')
plt.plot(threshold_curve, recall_curve[1:], label='recall', color = '#DEDCBB')
plt.legend(loc='lower left')
plt.xlabel('Threshold (above this probability)');
plt.title('Precision and Recall Curves');


# In[93]:


y_predict = (Random_Forest.predict_proba(X_train)[:, 1] >= 0.66)

print("Default threshold:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_train, y_predict), 
                                                     recall_score(y_train, y_predict)))


# In[95]:


y_predict = (Random_Forest.predict_proba(X_train)[:, 1] >= 0.61)

loan_confusion = confusion_matrix(y_train, y_predict)

sns.heatmap(loan_confusion , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Train Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# In[96]:


# confusion matrix for crossval
y_pred = cross_val_predict(Random_Forest, X = X_train, y = y_train, cv=5)
conf_mat = confusion_matrix(y_train, y_pred)
sns.heatmap(conf_mat , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Cross Validation Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# ## Extra Tree
# ---

# In[97]:


Extra_Tree = ExtraTreesClassifier()
Extra_Tree.fit(X_train, y_train)
scores = cross_val_score(Extra_Tree, X_train, y_train, cv =5, scoring = 'f1')
print('f1_scores for validation: ',scores)
print('Mean f1_score for validation: ',scores.mean())


# In[98]:


y_predict = (Extra_Tree.predict_proba(X_train)[:, 1] >= 0.1)

print("Default threshold:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_train, y_predict), 
                                                     recall_score(y_train, y_predict)))


# In[99]:


# confusion matrix for crossval
y_pred = cross_val_predict(Extra_Tree, X = X_train, y = y_train, cv=5)
conf_mat = confusion_matrix(y_train, y_pred)
sns.heatmap(conf_mat , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Cross Validation Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# ## Stacking
# ---

# In[100]:


lr = LogisticRegression() 
stacked = StackingClassifier(classifiers =[knn,Decision_Tree, lr], meta_classifier = lr, use_probas = False)
model_stack = stacked.fit(X_train, y_train)   # training of stacked model
accuracies = cross_val_score(estimator = model_stack, X = X_train, y = y_train, cv = 5, scoring='f1')
print('f1_score stacking for cross validation : ',accuracies)
print('Mean f1_score stacking for cross validation : ',accuracies.mean())


# In[152]:


precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_train, stacked.predict_proba(X_train)[:,1] )
plt.plot(threshold_curve, precision_curve[1:],label='precision', color = '#724949')
plt.plot(threshold_curve, recall_curve[1:], label='recall', color = '#DEDCBB')
plt.legend(loc='lower left')
plt.xlabel('Threshold (above this probability)');
plt.title('Precision and Recall Curves');


# In[101]:


y_predict = (stacked.predict_proba(X_train)[:, 1] >= 0.1)

print("Default threshold:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_train, y_predict), 
                                                     recall_score(y_train, y_predict)))


# In[102]:


# confusion matrix for crossval
y_pred = cross_val_predict(stacked, X = X_train, y = y_train, cv=5)
conf_mat = confusion_matrix(y_train, y_pred)
sns.heatmap(conf_mat , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Cross Validation Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# ## Bagging
# ---

# In[106]:


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=50,
    max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)

accuracies = cross_val_score(estimator = bag_clf, X = X_train, y = y_train, cv = 5, scoring='f1')
print('f1_score Bagging for cross validation : ',accuracies)
print('Mean f1_score Bagging for cross validation : ',accuracies.mean())


# In[153]:


precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_train, bag_clf.predict_proba(X_train)[:,1] )
plt.plot(threshold_curve, precision_curve[1:],label='precision', color = '#724949')
plt.plot(threshold_curve, recall_curve[1:], label='recall', color = '#DEDCBB')
plt.legend(loc='lower left')
plt.xlabel('Threshold (above this probability)');
plt.title('Precision and Recall Curves');


# In[107]:


y_predict = (bag_clf.predict_proba(X_train)[:, 1] >= 0.1)

print("Default threshold:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_train, y_predict), 
                                                     recall_score(y_train, y_predict)))


# In[111]:


# confusion matrix for crossval
y_pred = cross_val_predict(bag_clf, X = X_train, y = y_train, cv=5)
conf_mat = confusion_matrix(y_train, y_pred)
conf_mat


# In[112]:


# confusion matrix for crossval
y_pred = cross_val_predict(bag_clf, X = X_train, y = y_train, cv=5)
conf_mat = confusion_matrix(y_train, y_pred)
sns.heatmap(conf_mat , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Cross Validation Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# ## Boosting
# ---

# * ### AdaBoost

# In[110]:


param_grid = {'base_estimator__criterion' : ['gini', 'entropy'],
              'base_estimator__splitter' :   ['best', 'random'],
              'n_estimators': [1, 5, 10, 20, 100, 500]
             }
DTC = DecisionTreeClassifier(random_state = 0)
ABC = AdaBoostClassifier(base_estimator = DTC)

# run grid search
grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring = 'f1')
grid_search_ABC.fit(X_train, y_train)

print('\n Best param after grid search', grid_search_ABC.best_params_)
print('\n Best score after grid search', grid_search_ABC.best_score_)


# In[154]:


precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_train, grid_search_ABC.predict_proba(X_train)[:,1] )
plt.plot(threshold_curve, precision_curve[1:],label='precision', color = '#724949')
plt.plot(threshold_curve, recall_curve[1:], label='recall', color = '#DEDCBB')
plt.legend(loc='lower left')
plt.xlabel('Threshold (above this probability)');
plt.title('Precision and Recall Curves');


# In[113]:


y_predict = (grid_search_ABC.predict_proba(X_train)[:, 1] >= 0.1)

print("Default threshold:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_train, y_predict), 
                                                     recall_score(y_train, y_predict)))


# In[114]:


# confusion matrix for crossval
y_pred = cross_val_predict(grid_search_ABC, X = X_train, y = y_train, cv=5)
conf_mat = confusion_matrix(y_train, y_pred)
sns.heatmap(conf_mat , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Cross Validation Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# * ### Gradient Boosting

# In[119]:


gbc = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.05)
gbc.fit(X_train, y_train)
accuracies = cross_val_score(estimator = gbc, X = X_train, y = y_train, cv = 5, scoring='f1')
print('f1_score Gradient Boosting for cross validation : ',accuracies)
print('Mean f1_score Gradient Boosting for cross validation : ',accuracies.mean())


# In[155]:


precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_train, gbc.predict_proba(X_train)[:,1] )
plt.plot(threshold_curve, precision_curve[1:],label='precision', color = '#724949')
plt.plot(threshold_curve, recall_curve[1:], label='recall', color = '#DEDCBB')
plt.legend(loc='lower left')
plt.xlabel('Threshold (above this probability)');
plt.title('Precision and Recall Curves');


# In[120]:


y_predict = (gbc.predict_proba(X_train)[:, 1] >= 0.1)

print("Default threshold:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_train, y_predict), 
                                                     recall_score(y_train, y_predict)))


# In[121]:


# confusion matrix for crossval
y_pred = cross_val_predict(gbc, X = X_train, y = y_train, cv=5)
conf_mat = confusion_matrix(y_train, y_pred)
sns.heatmap(conf_mat , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Cross Validation Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# In[ ]:


# # A sample parameter
# parameters = {
#     'loss':['deviance'],
#     'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     'min_samples_split': np.linspace(0.1, 0.5, 12),
#     'min_samples_leaf': np.linspace(0.1, 0.5, 12),
#     'max_depth':[1,3,5],
#     'max_features':['log2','sqrt'],
#     'criterion': ['friedman_mse',  'mae'],
#     'subsample':[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     'n_estimators':[5]
#     }


# #passing the scoring function in the GridSearchCV
# clf = GridSearchCV(GradientBoostingClassifier(), parameters, scoring='f1',refit=False,cv=5, n_jobs=-1)
# clf.fit(X_train, y_train)

# print('\n Best param after grid search', clf.best_params_)
# print('\n Best score after grid search', clf.best_score_)


# * ### XGBoost

# In[118]:


X_train.columns = X_train.columns.str.replace('<','less').str.replace(' ','_')

xgboost = XGBClassifier(n_estimators = 100, learning_rate = 0.05)
xgboost.fit(X_train, y_train)
accuracies = cross_val_score(estimator = xgboost, X = X_train, y = y_train, cv = 5, scoring='f1')
print('f1_score XGBoost for cross validation : ',accuracies)
print('Mean f1_score XGBoost for cross validation : ',accuracies.mean())


# In[150]:


precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_train, xgboost.predict_proba(X_train)[:,1] )
plt.plot(threshold_curve, precision_curve[1:],label='precision', color = '#724949')
plt.plot(threshold_curve, recall_curve[1:], label='recall', color = '#DEDCBB')
plt.legend(loc='lower left')
plt.xlabel('Threshold (above this probability)');
plt.title('Precision and Recall Curves');


# In[122]:


y_predict = (xgboost.predict_proba(X_train)[:, 1] >= 0.1)

print("Default threshold:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_train, y_predict), 
                                                     recall_score(y_train, y_predict)))


# In[123]:


# confusion matrix for crossval
y_pred = cross_val_predict(xgboost, X = X_train, y = y_train, cv=5)
conf_mat = confusion_matrix(y_train, y_pred)
sns.heatmap(conf_mat , cmap = 'pink_r', annot = True , square = True , fmt = 'd',
           xticklabels = ['long term','short term'],
           yticklabels = ['long term','short term'])
plt.title('Cross Validation Confusion Matrix',fontsize = 15)
plt.xlabel('prediction')
plt.ylabel('actual');


# In[ ]:


# X_train.columns = X_train.columns.str.replace('<','less').str.replace(' ','_')

# estimator = XGBClassifier(
#     objective= 'binary:logistic',
#     nthread=4,
#     seed=42)

# parameters = {
#     'max_depth': range (2, 10, 1),
#     'n_estimators': range(60, 220, 40),
#     'learning_rate': [0.1, 0.01, 0.05]}

# grid_search = GridSearchCV(estimator=estimator,param_grid=parameters,scoring = 'f1',n_jobs = 10,cv = 10,
#                            verbose=True)
# grid_search.fit(X_train, y_train)

# print('score',grid_search.best_estimator_)


# ## Voting Classifer (HARD)
# ---

# In[125]:


log=LogisticRegression() 
rnd=RandomForestClassifier()
dct=DecisionTreeClassifier()
voting_classifer = VotingClassifier(estimators=[('lr',log),('rf',rnd),('dt',dct)],voting='hard',n_jobs=-1)
voting_classifer.fit(X_train, y_train)
scores = cross_val_score(voting_classifer, X_train, y_train, cv=5, scoring='f1')
print('f1_score Voting Classifer for cross validation : ',scores)
print('Mean f1_score Voting Classifer for cross validation : ',scores.mean())


# In[ ]:


# models = [('rf', gd_sr1 ), ('ad', grid_search_ABC )]
# vc_wtd = VotingClassifier(estimators= models, voting='hard', n_jobs=-1)
# vc_wtd.fit(X_train, y_train)
# scores = cross_val_score(vc_wtd, X_train, y_train, cv=10, scoring='f1')
# print('lass weights Random Forest Valdition F1:',scores.mean())


# ___
# # Test
# ___

# In[148]:


# X_train.columns = X_train.columns.str.replace('<','less').str.replace(' ','_')
# xgboost = XGBClassifier(n_estimators = 100, learning_rate = 0.05)
# xgboost.fit(X_train, y_train)
# accuracies = cross_val_score(estimator = xgboost, X = X_train, y = y_train, cv = 5, scoring='f1')
# print('f1_score XGBoost for cross validation : ',accuracies)
# print('Mean f1_score XGBoost for cross validation : ',accuracies.mean())


# In[149]:


y_pred_ =xgboost.predict(X_test)
y_pred_1 =xgboost.predict(X_train)
scores_1 = metrics.f1_score(y_train, y_pred_1)
scores = metrics.f1_score(y_test, y_pred_)
print('Test score for XGBoost: ', scores)


# # Conclusion
# In Conclusion, after examen multiple classification models, it is clear that the best model to predict whether a loan is a short term or a long term is XGBoost classification model, which gave the highest cross validation F1. The last step for this project is to train the best model using cross validation sets (80% of the data), and test it using the splatted data for testing (20% of the data set). Then F1 score for both the training and testing will be printed. F1 for the cross validition set is 0.8731 and for the testing set 0.8691 as shown above. However, For future work, more classification models will be examined.
