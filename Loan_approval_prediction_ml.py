#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data manipulation & handling libraries
import pandas as pd 
import numpy as np 

# Data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sci

# VIF library
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Data preprocessing libraries
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer,KNNImputer

# Model selection libraries
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score

# Machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
import xgboost

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
from sklearn.naive_bayes import GaussianNB

# Hyperparameter tuning parameters
from sklearn.model_selection import RandomizedSearchCV

# Clustering
from sklearn.cluster import KMeans

# Feature importance library
from sklearn.feature_selection import RFE

# Learning curve analysis
from sklearn.model_selection import learning_curve

# Deep learning libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')


# ## Loading the dataset

# In[2]:


df = pd.read_excel('Rocket_Loans.xlsx')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.duplicated().sum()


# In[6]:


df.drop('Loan_ID',axis=1,inplace=True)


# In[7]:


df.head()


# In[8]:


df.columns


# In[9]:


len(df.columns)


# ## Problem statement: 
#     - To predict loan approval (1 or 0) based on applicant's details

# ## Studying the dataset

# In[10]:


df.info()


# ### Analysis:
#     - From the result it can be seen that there are a few null values in sex(13),married(3),no.of people in 
#     family(15),self_employed(32),amount_disbursed(21),loan_tenure(14),credit_score(50) columns

# In[11]:


df.describe(include=[np.number])


# ### Analysis:
#     1. The age feature can be considered as a normal distribution since mean is equal to median
#     2. Loan_bearer_income,loan_cobearer_income are highly positively skewed since mean is greater than median and there may be outliers beyond the upper whisker region
#     3. Amount_disbursed is highly positively skewed with outliers beyond the upper whisker
#     4. loan_tenure is moderately negatively skewed with a few outliers in the lower whisker region

# In[12]:


df.describe(include=[np.object])


# ### Analysis:
#     - The target variable being loan_status is an imbalanced categorical variable

# In[13]:


for i in df.columns:
    print(f'The number of unique values in {i} is {df[i].nunique()}')
    if(df[i].nunique() <= 10):
        print(df[i].value_counts())


# ### Analysis:
#     - As seen sex,married,no. of people in family,qualification,self_employed,loan_tenure,credit_score,location_type,
#     loan_status can be considered as categorical variables

# ###  loan approval rate for applicants having credit score

# In[14]:


round((df.groupby('Credit_Score')['Loan_Status'].count().sort_values(ascending=False)[1]/df['Credit_Score'].count())*100,2)


# ### frequency distribution of total income
# (i.e. Total Income = Loan_bearer_income + Loan_Cobearer_income)
#  - Below 5k
#  - 5k to 10k
#  - 10k to 15k
#  - Above 15k

# In[15]:


df1 = pd.read_excel('Rocket_Loans.xlsx')


# In[16]:


df1


# In[17]:


df1['Loan_Bearer_Income'] + df1['Loan_Cobearer_Income'].value_counts()


# In[18]:


df1['freq'] = pd.cut(x=df1['Loan_Bearer_Income'] + df1['Loan_Cobearer_Income'],
                  bins=[0,5000,10000,15000,41667],include_lowest=True,labels=['Below 5k','5k to 10k','10k to 15k','Above 15k'])


# In[19]:


freq_table = pd.crosstab(df1['freq'],'frequency')


# In[20]:


freq_table


# ## Exploratory Descriptive Analysis

# ### 1. Custom descriptive statistics function

# In[21]:


def num_custom_summary(data):
    result = []
    from collections import OrderedDict
    
    for i in data.columns:
        stats = OrderedDict({'Column name': i,
                             'Data type':data[i].dtype,
                            'Count':data[i].notnull().count(),
                            'Non-null values':data[i].notnull().sum(),
                            'Null values':data[i].isnull().sum(),
                            'Minimum':data[i].min(),
                            'Q1':data[i].quantile(0.25),
                            'Mean':data[i].mean(),
                            'Q2':data[i].quantile(0.5),
                            'Q3':data[i].quantile(0.75),
                            'Maximum':data[i].max(),
                             'Variance':data[i].var(),
                             'Std Dev':data[i].std(),
                            'Kurtosis':data[i].kurt(),
                            'Skewness':data[i].skew(),
                            'IQR':data[i].quantile(0.75) - data[i].quantile(0.25)})
        result.append(stats)
        
        # Labels for skewness
        if data[i].skew() >= 1:
            slabel = 'Highly positively skewed'
        elif 0.5 <= data[i].skew() < 1:
            slabel = 'Moderately positively skewed'
        elif 0 <= data[i].skew() < 0.5:
            slabel = 'Fairly Symmetric(positive)'
        elif -0.5 <= data[i].skew() < 0:
            slabel = 'Fairly symmetric(negative)'
        elif -1 <= data[i].skew() < -0.5:
            slabel = 'Moderately negatively skewed'
        elif data[i].skew() <= -1:
            slabel = 'Highly negatively skewed'
        else:
            slabel = 'Error'
        stats['Skewness comments'] = slabel
        
        # Labels for outliers
        upper_whisker = stats['Q3'] + stats['IQR']*1.5
        lower_whisker = stats['Q1'] - stats['IQR']*1.5
        if len([x for x in data[i] if x < lower_whisker or x > upper_whisker]) > 0:
            olabel = 'Has outliers'
        else:
            olabel = 'No outliers'
        
        stats['Outlier comments'] = olabel
        stats['No. of outliers'] = len((data.loc[(data[i]< lower_whisker) | (data[i]> upper_whisker)]))
        
        result_df = pd.DataFrame(data=result)
    result_df = result_df.T
    result_df.rename(columns=result_df.iloc[0, :], inplace=True) 
    result_df.drop(result_df.index[0], inplace=True)
    return result_df


# In[22]:


num_df = df[['Age','Loan_Bearer_Income','Loan_Cobearer_Income','Amount Disbursed']]
cat_df = df[['Sex','Married','No. of People in the Family','Qualification','Self_Employed','Loan_Tenure',
             'Credit_Score','Location_type','Loan_Status']]


# In[23]:


num_custom_summary(num_df)


# #### Analysis:
#     1. The Amount Disbursed feature is the only one with null values 
#     2. Loan_Bearer_Income,Loan_Cobearer_Income,Amount Disbursed contain sufficient amount of outliers and are highly positively skewed

# In[24]:


def cat_custom_summary(data):
    from collections import OrderedDict
    result =[]
    
    for i in data.columns:
        stats = OrderedDict({'Column name':i,
                            'Data type':data[i].dtype,
                            'Count':data[i].notnull().count(),
                            'Non-null values':data[i].notnull().sum(),
                            'Null values':data[i].isnull().sum(),
                            'No. of unique values':data[i].nunique(),
                            'Category with highest records':data[i].value_counts().idxmax(),
                             'Highest no. of records':data[i].value_counts().max(),
                            'Category with lowest records':data[i].value_counts().idxmin(),
                             'Lowest no. of records':data[i].value_counts().min(),
                            })
        result.append(stats)
    result_df = pd.DataFrame(data=result)
    result_df = result_df.T
    result_df.rename(columns=result_df.iloc[0, :], inplace=True) 
    result_df.drop(result_df.index[0], inplace=True)
    return result_df


# In[25]:


cat_custom_summary(cat_df)


# #### Analysis:
#     1. Except Qualification,Location_type and Loan_Status the rest of the categorical features have null values

# ### 2. Dealing with missing values

# In[26]:


df.isnull().sum()


# #### Analysis:
#     1. Since there is no feature with more than 30% null values there is no need to delete any of the features
#     2. Any form of imputation method will be suffice for missing value treatment

# In[27]:


null_data = df[df.isnull().any(axis=1)]
a = null_data.isnull().sum(axis=1).tolist()


# In[28]:


j = 0
for i in range (0,len(a)):
    if a[i] >= 10:
        print('The {i} row has most values missing')
        j = i+1
if j == 0:
    print(f'There is no row amoung {null_data.shape[0]} rows that has more than 75% values missing')


# #### Analysis:
#     1. Hence we don't need to consider deleting any rows

# In[29]:


fig = plt.figure(figsize = (15,18))

ax = fig.add_subplot(3,3,1) 
sns.countplot(x='Loan_Status',hue='Married',data=df)
ax.title.set_text('Loan_Status vs. Married')


ax = fig.add_subplot(3,3,2)
sns.countplot(x='Loan_Status',hue='Sex',data=df)
ax.title.set_text('Loan_Status vs. Sex')

ax = fig.add_subplot(3,3,3) 
sns.countplot(x='Loan_Status',hue='No. of People in the Family',data=df)
ax.title.set_text('Loan_Status vs. No. of people in family')

ax = fig.add_subplot(3,3,4)
sns.countplot(x='Loan_Status',hue='Qualification',data=df)
ax.title.set_text('Loan_Status vs. Qualification')

ax = fig.add_subplot(3,3,5)
sns.countplot(x='Loan_Status',hue='Self_Employed',data=df)
ax.title.set_text('Loan_Status vs. Self Employed')

ax = fig.add_subplot(3,3,6)
sns.countplot(x='Loan_Status',hue='Loan_Tenure',data=df)
ax.title.set_text('Loan_Status vs. Loan_Tenure')

plt.show()


# ### 2.1. Using multivariate feature imputation

# In[30]:


num_df = df[['Age','Loan_Bearer_Income','Loan_Cobearer_Income','Amount Disbursed']]
cat_df = df[['Sex','Married','No. of People in the Family','Qualification','Self_Employed','Loan_Tenure',
             'Credit_Score','Location_type','Loan_Status']]


# In[31]:


imp2 = IterativeImputer(max_iter=10,random_state=10)


# In[32]:


num_df2 = pd.DataFrame(imp2.fit_transform(num_df),columns=['Age','Loan_Bearer_Income',
                                                           'Loan_Cobearer_Income','Amount Disbursed'])


# In[33]:


num_df2


# In[105]:


num_df.mean()


# In[34]:


ss = StandardScaler()


# In[35]:


num_df2 = pd.DataFrame(ss.fit_transform(num_df2),columns=num_df2.columns)


# In[36]:


num_df2


# In[37]:


num_df2.isnull().sum()


# In[38]:


X = df[['Sex','Married','No. of People in the Family','Qualification','Self_Employed','Loan_Tenure',
             'Credit_Score','Location_type','Loan_Status']]
result =[] 
le = LabelEncoder()
for j in X.columns:
    fit_by = pd.Series([i for i in X[j].unique() if type(i) == str])
    le.fit(fit_by)
    ### Set transformed col leaving np.NaN as they are
    X[j] = X[j].apply(lambda x: le.transform([x])[0] if type(x) == str else x)
    result.append(X[j])
result_df = pd.DataFrame(data=result)
result_df = result_df.T


# In[39]:


result_df.isnull().sum()


# In[40]:


cat_df2 = pd.DataFrame(imp2.fit_transform(result_df),columns=['Sex','Married','No. of People in the Family',
                                                              'Qualification','Self_Employed','Loan_Tenure',
                                                              'Credit_Score','Location_type','Loan_Status'])


# In[41]:


cat_df2.isnull().sum()


# In[42]:


cat_df2['Loan_Tenure'] = cat_df2['Loan_Tenure'].replace({12.0:1,36.0:2,60.0:3,84.0:4,120.0:5,180.0:6,240.0:7,300.0:8,360.0:9,480.0:10}).astype(int)


# In[43]:


df2 = cat_df2.join(num_df2)


# In[44]:


df2.head()


# In[109]:


df.head()


# ### 3. Dealing with outliers

# In[45]:


df.boxplot(figsize= (10,6))


# #### Analysis:
#     1. Loan_Bearer_Income,Loan_Cobearer_Income,Amount Disbursed are the continuous features that contain outliers
#     2. Since Loan_Tenure and Credit_Score are considered as categorical features outliers are not considered
#     3. There are no outliers in Age feature

# In[46]:


df['No. of People in the Family'].value_counts()


# In[47]:


fig = plt.figure(figsize = (15,20))

ax = fig.add_subplot(4,3,1) 
sns.histplot(df['Sex'])
ax.title.set_text('Sex')


ax = fig.add_subplot(4,3,2)
sns.histplot(df['Age'])
ax.title.set_text('Age')

ax = fig.add_subplot(4,3,3) 
sns.histplot(df['Married'])
ax.title.set_text('Married')

ax = fig.add_subplot(4,3,4)
sns.histplot(df['Qualification'])
ax.title.set_text('Qualification')

ax = fig.add_subplot(4,3,5)
sns.histplot(df['Self_Employed'])
ax.title.set_text('Self_Employed')

ax = fig.add_subplot(4,3,6)
sns.histplot(df['Loan_Bearer_Income'])
ax.title.set_text('Loan_Bearer_Income')

ax = fig.add_subplot(4,3,7)
sns.histplot(df['Loan_Cobearer_Income'])
ax.title.set_text('Loan_Cobearer_Income')

ax = fig.add_subplot(4,3,8)
sns.histplot(df['Amount Disbursed'])
ax.title.set_text('Amount Disbursed')

ax = fig.add_subplot(4,3,9)
sns.histplot(df['Loan_Tenure'])
ax.title.set_text('Loan_Tenure')

ax = fig.add_subplot(4,3,10)
sns.histplot(df['Credit_Score'])
ax.title.set_text('Credit_Score')

ax = fig.add_subplot(4,3,11)
sns.histplot(df['Location_type'])
ax.title.set_text('Location_type')

ax = fig.add_subplot(4,3,12)
sns.histplot(df['Loan_Status'])
ax.title.set_text('Loan_Status') 

plt.show()


# #### Analysis:
#     1. None of the categorical features contain any outliers
#     2. Loan_Bearer_Income,Loan_Cobearer_Income,Amount Disbursed have right tailed skewness or are positively skewed

# In[48]:


for i in num_df2.columns:
    Q1 = num_df2[i].quantile(0.25)
    Q3 = num_df2[i].quantile(0.75)
    IQR = Q3-Q1
    upper_whisker = Q3 + IQR *1.5
    lower_whisker = Q1 - IQR *1.5
    print(f'{i} has {len((num_df2.loc[(num_df2[i]< lower_whisker) | (num_df2[i]> upper_whisker)]))} outliers')


# #### Applying log transformation

# In[49]:


num_df = df[['Age','Loan_Bearer_Income','Loan_Cobearer_Income','Amount Disbursed']]
cat_df = df[['Sex','Married','No. of People in the Family','Qualification','Self_Employed','Loan_Tenure',
             'Credit_Score','Location_type','Loan_Status']]


# In[50]:


for i in ['Loan_Bearer_Income','Loan_Cobearer_Income','Amount Disbursed']:   
    num_df2[i] = np.log(num_df2[i])       


# In[51]:


for i in num_df2.columns:
    Q1 = num_df2[i].quantile(0.25)
    Q3 = num_df2[i].quantile(0.75)
    IQR = Q3-Q1
    upper_whisker = Q3 + IQR *1.5
    lower_whisker = Q1 - IQR *1.5
    print(f'{i} has {len((num_df2.loc[(num_df2[i]< lower_whisker) | (num_df2[i]> upper_whisker)]))} outliers')


# In[52]:


sns.histplot(num_df2['Loan_Bearer_Income'])


# ## Multicollinearity test

# ### Stage 1: Correlation heatmap

# In[53]:


corr = df2.corr()
plt.subplots(figsize = (10,10))
sns.heatmap(corr,annot=True)


# #### Analysis:
# 1. Married and Sex has 36% correlation
# 2. Married and No. of People in the Family has 33% correlation
# 3. Loan_Bearer_Income and Amount Disbursed has 57% correlation
# 
# -Conclusion: Some features have correlation greater than 30%. So we can conclude collinearity exists as stage 1 results

# ### Stage 2: Variance Inflation Factor(VIF)

# In[54]:


def VIF(features):
    vif = pd.DataFrame()
    vif['VIF Score'] = [variance_inflation_factor(features.values,i) for i in range(features.shape[1])]
    vif['Features']  = features.columns
    vif.sort_values(by=['VIF Score'],ascending= False,inplace=True)
    return vif


# In[55]:


df2.columns


# In[56]:


VIF(df2.drop('Loan_Status',axis=1))


# #### Analysis:
#     Loan_Tenure,Credit_Score,Sex have VIF greater than 5 so we can conclude multicollinearity exists as per stage 2 results

# ### Correlation with target feature

# In[57]:


def cwt(data,t_col):
    independent_variables = data.drop(t_col,axis = 1).columns
    corr_result =[]
    for col in independent_variables:
        corr_result.append(data[t_col].corr(data[col]))
    result = pd.DataFrame([independent_variables,corr_result],index=['Independent_variables','Correlation']).T
    return result.sort_values(by= ['Correlation'])


# In[58]:


cwt(df2,'Loan_Status')


# #### Analysis:
#     Only Credit_Score is highly correlated with Loan_Status(54%)

# ## Applying PCA to treat multicollinearity

# In[59]:


def pca_func(X):
    n_comp = len(X.columns)
    pcs =1
    # Feature scaling
    X = StandardScaler().fit_transform(X)
    
    # Applying PCA
    for i in range(1,n_comp):
        pca = PCA(n_components=i)
        p_comp = pca.fit_transform(X)
        evr = np.cumsum(pca.explained_variance_ratio_)
        if(evr[i-1]>0.9):
            pcs = i
            break
            
    print('Explained variance ratio after PCA is:',evr)
#creating dataframe using principal components
    col = []
    for j in range(1,pcs+1):
        col.append('PC_'+str(j))
    pca_df = pd.DataFrame(data=p_comp,columns=col)   
    return pca_df


# In[60]:


pca_df2 = pca_func(df2.drop('Loan_Status',axis=1))


# In[61]:


pca_df2.head()


# ### Joining PCA feature with target feature

# In[62]:


transformed_df2 = pca_df2.join(df2['Loan_Status'])


# In[63]:


transformed_df2.head()


# ## Model Building - Machine Learning models

# ### Train test split

# In[64]:


def train_and_test_split(data,y,test_size=0.2,random_state=10):
    X = data.drop(y,1)
    return train_test_split(X,data[y],test_size=test_size,random_state=random_state,shuffle=True,stratify=data[y])


# In[65]:


def model_builder(model_name,estimator,data,t_col):
    X_train,X_test,y_train,y_test = train_and_test_split(data,t_col)
    estimator.fit(X_train,y_train)
    y_pred = estimator.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    fscore = f1_score(y_test,y_pred)
    return model_name,accuracy,fscore


# ### Building multiple models

# In[66]:


def multiple_models(data,data1,t_col):
    col = ['Model_Name','Accuracy_Score','F1_Score']
    result = pd.DataFrame(columns=col)
    
    # Adding values to result dataframe
    result.loc[len(result)] = model_builder('Logistic regression',LogisticRegression(),data1,t_col)
    result.loc[len(result)] = model_builder('Decision Tree classifer',DecisionTreeClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('Gaussian Naive Bayes',GaussianNB(),data1,t_col)
    result.loc[len(result)] = model_builder('Support Vector classifer',SVC(),data1,t_col)
    result.loc[len(result)] = model_builder('K Neighbors classifer',KNeighborsClassifier(),data1,t_col)
    result.loc[len(result)] = model_builder('Random Forest classifer',RandomForestClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('Adaboost classifer',AdaBoostClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('Gradient Boost classifer',GradientBoostingClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('XGBoost classifer',XGBClassifier(verbosity = 0),data,t_col)
    
    return result.sort_values(by=['F1_Score'],ascending=False,ignore_index=True)


# In[67]:


multiple_models(df2,transformed_df2,'Loan_Status')


# ## Cross Validation

# In[68]:


def kfold_cv(data,data1,t_col,cv=10):
    model_names = [LogisticRegression(),SVC(),KNeighborsClassifier(),GaussianNB(),DecisionTreeClassifier(),
                   RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),
                   XGBClassifier(verbosity = 0)]
    accscores = ['Score_LR','Score_SVC','Score_KNC','Score_GNB','Score_DTC','Score_RFC',
             'Score_ABC','Score_GBC','Score_XGBC']
    fscores = ['Score_LR','Score_SVC','Score_KNC','Score_GNB','Score_DTC','Score_RFC',
             'Score_ABC','Score_GBC','Score_XGBC']
    stf = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    
    for model,i in zip(model_names,range(len(accscores))):
        if(i<=2):
            accscores[i] = (cross_val_score(estimator=model,X=data1.drop(t_col,1),y=data1[t_col],cv=stf))
            fscores[i] = (cross_val_score(estimator=model,X=data1.drop(t_col,1),y=data1[t_col],cv=stf,scoring='f1'))
        else:
            accscores[i] = (cross_val_score(estimator=model,X=data.drop(t_col,1),y=data[t_col],cv=stf))
            fscores[i] = (cross_val_score(estimator=model,X=data.drop(t_col,1),y=data[t_col],cv=stf,scoring='f1'))
            
    
    result = []
    for i in range(len(model_names)):
        accscore_mean = np.mean(accscores[i])
        accscore_std = np.std(accscores[i])
        fscore_mean = np.mean(fscores[i])
        fscore_std = np.std(fscores[i])
        model_name = type(model_names[i]).__name__
        temp = [model_name,accscore_mean,accscore_std,fscore_mean,fscore_std]
        result.append(temp)
    
    result_df = pd.DataFrame(result,columns=['Model Name','Accuracy Score','Accuracy Std Dev','F1 Score','F1 Std Dev'])
    return result_df.sort_values(by=['F1 Score'],ascending=False,ignore_index=True)


# In[69]:


kfold_cv(df2,transformed_df2,'Loan_Status')


# ## Hyperparameter tuning

# In[63]:


def tuning(X,y,X1,y1,cv=10):
    # Creating the parameter grid
    
    param_knn = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,20,30],'leaf_size':[1,2,3,4,5,6,7,8,9,10,20,30,40,50],'p':[1,2]}
    param_dtc = {'max_depth':[2, 3, 5, 10, 20],'min_samples_leaf':[5, 10, 20, 50, 100],
                 'criterion':["gini", "entropy"]}
    param_adb = {'learning_rate':[0.0001,0.001,0.01,0.1,1],'n_estimators': [10, 50, 100, 500]}
    param_svc = {'C': [0.1, 1, 10, 100, 1000],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

    param_gboost = {"n_estimators":[5,50,250,500],"max_depth":[1,3,5,7,9],"learning_rate":[0.01,0.1,1,10,100]}
    param_xgb = {'learning_rate': [0.001,0.01,0.1,1],'reg_alpha': [0.1,0.3,0.5,0.7,0.9,1],
                 'reg_lambda': [0.1,0.3,0.5,0.7,0.9,1]}
    param_rf = {'n_estimators':[50,100,150,200,250,300],
               'criterion':["gini", "entropy"]}
    param_gnb = {'var_smoothing': np.logspace(0,-9, num=100)}
    
    # Hyperparameter tuning
    stf = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    tune_knn = RandomizedSearchCV(KNeighborsClassifier(),param_knn,cv=stf)  
    tune_dtc = RandomizedSearchCV(DecisionTreeClassifier(),param_dtc,cv=stf)    
    tune_adb = RandomizedSearchCV(AdaBoostClassifier(),param_adb,cv=stf)    
    tune_svc = RandomizedSearchCV(SVC(),param_svc,cv=stf)    
    tune_gboost = RandomizedSearchCV(GradientBoostingClassifier(),param_gboost,cv=stf)    
    tune_xgb = RandomizedSearchCV(XGBClassifier(),param_xgb,cv=stf)    
    tune_rf = RandomizedSearchCV(RandomForestClassifier(),param_rf,cv=stf) 
    tune_gnb = RandomizedSearchCV(GaussianNB(),param_gnb,cv=stf) 
    
    
    # Model fitting
    tune_models = [tune_knn,tune_dtc,tune_gnb,tune_adb,tune_svc,tune_gboost,tune_xgb,tune_rf]
    models = ['KNN','DTC','GNB','ADB','SVC','GBoost','XGB','RF']
    for i in range(len(tune_models)):
        if(i<=0):
            tune_models[i].fit(X1,y1)
        else:
            tune_models[i].fit(X,y)
        
    for i in range(len(tune_models)):
        print('Model: ',models[i])
        print('Best parameters: ',tune_models[i].best_params_)


# In[64]:


tuning(df2.drop('Loan_Status',axis=1),df2['Loan_Status'],transformed_df2.drop('Loan_Status',axis=1),transformed_df2['Loan_Status'])


# In[70]:


def CV_post_hpt(X,y,X1,y1,cv=10):
    stf = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    accscore_lr = cross_val_score(LogisticRegression(),X1,y1,cv=stf)
    accscore_knn = cross_val_score(KNeighborsClassifier(n_neighbors=7,leaf_size=9,p=1),X1,y1,cv=stf)
    accscore_dtc = cross_val_score(DecisionTreeClassifier(criterion='entropy',min_samples_leaf=50,
                                                       max_depth=20),X,y,cv=stf)
    accscore_svc = cross_val_score(SVC(C=1,gamma=0.01),X1,y1,cv=stf)
    accscore_rf = cross_val_score(RandomForestClassifier(n_estimators=250,criterion='entropy'),X,y,cv=stf)
    accscore_adb = cross_val_score(AdaBoostClassifier(learning_rate=0.01,n_estimators=10),X,y,cv=stf)
    accscore_gboost = cross_val_score(GradientBoostingClassifier(n_estimators=5,max_depth=1,learning_rate=1),X,y,cv=stf)
    accscore_xgb = cross_val_score(XGBClassifier(learning_rate=0.001,reg_alpha=0.9,reg_lambda=0.1),X,y,cv=stf)
    accscore_gnb = cross_val_score(GaussianNB(var_smoothing=1.519911082952933e-06),X,y,cv=stf)
    
    fscore_lr = cross_val_score(LogisticRegression(),X1,y1,cv=stf,scoring='f1')
    fscore_knn = cross_val_score(KNeighborsClassifier(n_neighbors=7,leaf_size=9,p=1),X1,y1,cv=stf,scoring='f1')
    fscore_dtc = cross_val_score(DecisionTreeClassifier(criterion='entropy',min_samples_leaf=50,
                                                       max_depth=20),X,y,cv=stf,scoring='f1')
    fscore_svc = cross_val_score(SVC(C=1,gamma=0.01),X1,y1,cv=stf,scoring='f1')
    fscore_rf = cross_val_score(RandomForestClassifier(n_estimators=250,criterion='entropy'),X,y,cv=stf,scoring='f1')
    fscore_adb = cross_val_score(AdaBoostClassifier(learning_rate=0.01,n_estimators=10),X,y,cv=stf,scoring='f1')
    fscore_gboost = cross_val_score(GradientBoostingClassifier(n_estimators=5,max_depth=1,learning_rate=1),X,y,cv=stf,
                                    scoring='f1')
    fscore_xgb = cross_val_score(XGBClassifier(learning_rate=0.001,reg_alpha=0.9,reg_lambda=0.1),X,y,cv=stf,
                                 scoring='f1')
    fscore_gnb = cross_val_score(GaussianNB(var_smoothing=1.519911082952933e-06),X,y,cv=stf,scoring='f1')
    
    
    model_names = ['Logistic Regression','K Neighbors Classifier','Decision Tree Classifier',
                   'Support Vector Classifier','Random Forest Classifier','AdaBoost Classifier',
                   'Gradient Boosting Classifier','XGB Classifier','Gaussian Naive Bayes']
    accscores = [accscore_lr,accscore_knn,accscore_dtc,accscore_svc,accscore_rf,accscore_adb,accscore_gboost,
                 accscore_xgb,accscore_gnb]
    fscores = [fscore_lr,fscore_knn,fscore_dtc,fscore_svc,fscore_rf,fscore_adb,fscore_gboost,
                 fscore_xgb,fscore_gnb]
    
    result = []
    for i in range(len(model_names)):
        accscore_mean = np.mean(accscores[i])
        accscore_std = np.std(accscores[i])
        fscore_mean = np.mean(fscores[i])
        fscore_std = np.std(fscores[i])
        m_name = model_names[i]
        temp = [m_name,accscore_mean,accscore_std,fscore_mean,fscore_std]
        result.append(temp)
    result_df = pd.DataFrame(result,columns=['Model Name','Accuracy Score','Accuracy Std Dev','F1 Score','F1 Std Dev'])
    return result_df.sort_values(by='F1 Score',ascending=False,ignore_index=True)


# In[71]:


CV_post_hpt(df2.drop('Loan_Status',axis=1),df2['Loan_Status'],transformed_df2.drop('Loan_Status',axis=1),transformed_df2['Loan_Status'])


# ## Clustering

# In[72]:


labels = KMeans(n_clusters = 2,random_state = 10)
cluster = labels.fit_predict(df2.drop('Loan_Status',axis=1))


# In[73]:


def clustering(x,t_col,cluster):
    column = list(set(list(x.columns)) - set(x[t_col]))
    r = int(len(column)/2)
    if(r%2==0):
        r=r
    else:
        r+=1
    f,ax = plt.subplots(r,2,figsize=(20,18))
    a=0
    for row in range(r):
        for col in range(2):
            if(a!=len(column)):
                ax[row][col].scatter(x[t_col],x[column[a]],c=cluster)
                ax[row][col].set_xlabel(t_col)               
                ax[row][col].set_ylabel(column[a])               
                a+=1


# In[74]:


X = df2.drop('Loan_Status',axis=1)


# In[75]:


for col in X.columns:
    clustering(X,col,cluster)


# #### Analysis:
#     Loan_Tenure forms clusters with all features

# In[76]:


new_df2 = df2.join(pd.DataFrame(cluster,columns=['clusters']),how='left')
new_df2.head()


# In[77]:


temp_df2 = new_df2.groupby('clusters')['Loan_Status'].agg(['mean','median'])
temp_df2.head()


# In[78]:


cluster_df2 = new_df2.merge(temp_df2,on= 'clusters',how='left')
cluster_df2.head()


# In[79]:


X = cluster_df2.drop(['Loan_Status','clusters'],axis=1)
y = cluster_df2['Loan_Status']


# In[80]:


multiple_models(cluster_df2,cluster_df2,'Loan_Status')


# In[81]:


kfold_cv(cluster_df2,cluster_df2,'Loan_Status')


# In[82]:


CV_post_hpt(X,y,X,y)


# ## Feature importance using XGBoost

# In[83]:


X_train,X_test,y_train,y_test = train_and_test_split(cluster_df2.drop('clusters',axis=1),'Loan_Status')


# In[84]:


xgb = XGBClassifier()


# In[85]:


xgb.fit(X_train,y_train)


# In[86]:


xgboost.plot_importance(xgb)


# In[87]:


f_df2 = cluster_df2[['Loan_Bearer_Income','Amount Disbursed','Age','Loan_Cobearer_Income','Location_type',
                   'No. of People in the Family','Credit_Score','Loan_Status']]
f_df2.head()


# In[88]:


CV_post_hpt(f_df2.drop('Loan_Status',axis=1),f_df2['Loan_Status'],
            f_df2.drop('Loan_Status',axis=1),f_df2['Loan_Status'])


# ## RFE

# In[89]:


rfe = RFE(xgb)


# In[90]:


rfe.fit(X_train,y_train)


# In[91]:


rfe.support_


# In[92]:


X_train.columns


# In[93]:


rfe_df2 = cluster_df2[['Qualification','Self_Employed', 'Loan_Tenure', 'Credit_Score', 'Location_type', 
                       'Loan_Cobearer_Income', 'Amount Disbursed','Loan_Status']]
rfe_df2.head()


# In[94]:


CV_post_hpt(rfe_df2.drop('Loan_Status',1),rfe_df2['Loan_Status'],
            rfe_df2.drop('Loan_Status',1),rfe_df2['Loan_Status'])


# ## Learning curve analysis

# In[95]:


def generate_learning_curve(model_name,estimator,X,y,cv=10):
    train_size,train_score,test_score = learning_curve(estimator=estimator,X=X,y=y,cv=cv)
    train_score_mean = np.mean(train_score,axis=1)
    test_score_mean = np.mean(test_score,axis=1)
    plt.plot(train_size,train_score_mean,c='blue')
    plt.plot(train_size,test_score_mean,c='red')   
    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    plt.title('Learning curve for '+model_name)
    plt.legend(('Training_accuracy','Testing_accuracy'))


# In[96]:


model_names = [LogisticRegression(),SVC(),KNeighborsClassifier(),DecisionTreeClassifier(),
                   RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),
                   XGBClassifier(verbosity = 0)]
for a ,model in enumerate(model_names):
    fg = plt.figure(figsize=(12,15))
    ax = fg.add_subplot(5,2,a+1)
    generate_learning_curve(type(model_names[a]).__name__,model,rfe_df2.drop('Loan_Status',1),rfe_df2['Loan_Status'])


# ### Final model - Decision Tree Classifier

# In[170]:


classifier = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=50,max_depth=20)


# In[171]:


X_train,X_test,y_train,y_test = train_and_test_split(df2,'Loan_Status')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
fscore = f1_score(y_test,y_pred)


# In[103]:


# pickling the model
import pickle
pickle_out = open("loan_approval.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()





