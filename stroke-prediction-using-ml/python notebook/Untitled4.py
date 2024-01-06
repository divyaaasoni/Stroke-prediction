#!/usr/bin/env python
# coding: utf-8

# # Description:
# - According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient</apan>.

# ## Table of Contents:
# - Step 1 | Python Libraries
#     - 1.1 | Import Libraries
#     - 1.2 | Library configurations
# - Step 2 | Data
#     - 2.1 | Collecting Data
#     - 2.2 | Data Information
#     - 2.3 | Attribute Information
# - Step 3 | Data Preprocessning
#     - 3.1 | Missing Values Handling
#     - 3.2 | Visualization and Plots
#     - 3.3 | Plots Analysis
#     - 3.4 | Unique Values
#     - 3.5 | Normalization
# - Step 4 | Modeling
#     - 4.1 | Initialization
#     - 4.2 | RandomForestClassifier
#     - 4.3 | LogisticRegression
#     - 4.4 | SVC
#     - 4.5 | DecisionTreeClassifier
#     - 4.6 | KNeighborsClassifier
#     - 4.7 | result
#     - 4.8 | Final Modeling

# # Step1 | Python Libraries

# ## Step 1.1 | Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

################### Sklearn ####################################
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# ## Step 1.2 | Library configurations

# In[2]:


pd.options.mode.copy_on_write = True # Allow re-write on variable
sns.set_style('darkgrid') # Seaborn style
warnings.filterwarnings('ignore') # Ignore warnings
pd.set_option('display.max_columns', None) # Setting this option will print all collumns of a dataframe
pd.set_option('display.max_colwidth', None) # Setting this option will print all of the data in a feature


# # Step 2 | Data

# ## Step 2.1 | Collecting Data

# ### At first, import dataset in csv format by pandas library and read_csv method.

# In[3]:


data = pd.read_csv(r"C:\Users\hp\Downloads\archive (8)\healthcare-dataset-stroke-data.csv")
data.head()


# ## Step 2.2 | Data Informations

# ### We drop id columns, because its a unique identifier number.

# In[4]:


# Drop column = 'id'
data.drop(columns='id', inplace=True)
data.info()


# In[5]:


round(data.describe(include='all'), 2)


# ###  We have 5110 samples , with no null values

# ## Step 2.3 | Attribute Information

# #### Attribute Information
# |Feature|Describtion| |:-----:|:---------:| |id| unique identifier| |gender| "Male", "Female" or "Other"| |age| age of the patient| |hypertension| 0 if the patient doesn't have hypertension, 1 if the patient has hypertension| |heart_disease| 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart diseas| |ever_married| "No" or "Yes"| |work_type| "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"| |Residence_type| "Rural" or "Urban"| |avg_glucose_level| average glucose level in blood| |bmi| body mass index| |smoking_status| "formerly smoked", "never smoked", "smokes" or "Unknown"*| |stroke| 1 if the patient had a stroke or 0 if not|
# 
# Note : "Unknown" in smoking_status means that the information is unavailable for this patient.

# # Step 3 | Data Preprocessning

# ## Step 3.1 | Missing Values Handling

# In[6]:


data.isna().sum()


# In[7]:


print((data.isna().sum()/len(data))*100)


# ### There is 201 samples with no values in bmi column , its about 4% of data. For better result we drop them.

# In[8]:


data.dropna(how='any', inplace=True)


# ## Step 3.2 | Visualization and Plots

# In[9]:


cols = data.columns[:-1]
cols


# In[10]:


data


# In[11]:


numeric_columns = ['age', 'bmi', 'avg_glucose_level']
categorical_columns = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']


# In[12]:


i = 0
fig, ax = plt.subplots(3, 3, figsize=(15, 8))
plt.subplots_adjust(hspace = 0.5)
for num_col in numeric_columns :
    sns.kdeplot(x=num_col, hue='stroke', data=data, multiple='stack', ax=ax[i,0])
    sns.boxplot(x=num_col, data=data, ax=ax[i, 1])
    sns.scatterplot(x=num_col, y='stroke', data=data, ax=ax[i, 2])
    i+=1
plt.show()


# In[13]:


i=0
while i<8 :
    
    # Left AX
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(categorical_columns[i], size=20, weight='bold', color='navy')
    ax = sns.countplot(x=categorical_columns[i], data=data)
    ax.bar_label(ax.containers[0])
    ax.tick_params(axis='x', rotation=300)
    i+=1
    
    # Right AX
    plt.subplot(1, 2, 2)
    plt.title(categorical_columns[i], size=20, weight='bold', color='navy')
    ax = sns.countplot(x=categorical_columns[i], data=data)
    ax.bar_label(ax.containers[0])
    ax.tick_params(axis='x', rotation=300)
    i+=1
    plt.show()


# In[14]:


x = data['stroke'].value_counts()


explode = [0, 0.15]
labels = ['Stroke=0', 'Stroke=1']
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))

plt.pie(x, explode=explode, shadow=True, autopct='%1.1f%%', labels=labels, textprops=dict(color="w", weight='bold', size=15))
plt.legend()
plt.show()


# ## Step 3.3 | Plots Analysis

# ### Results :
# 1) About 96% of samples have not Stroke and 4% have stroke.
# 2) Distribution of samples is a Normal distribution.
# 3) Those who have had a stroke are in:
# age in range 40 to 85
# bmi in range 20 to 40
# glocuse level in range 50 to 130
# 4) About 60% of samples are female.
# 5) About 91% of samples dont have any hypertension.
# 6) About 95% of samples dont have any heart disease.
# 7) About 34% of samples never get married.
# 8) Most of samples worked in private.
# 9) We dont have any information in smoking field for 1483 of sapmples.

# ## Step 3.4 | Unique Values

#  #### We count number of unique values in each categorical column, to change them with integer values. Here we use .unique() command.

# In[15]:


columns_temp = ['gender', 'ever_married', 'work_type', 'smoking_status', 'Residence_type']

for col in columns_temp :
    print('column :', col)
    for index, unique in enumerate(data[col].unique()) :
        print(unique, ':', index)
    print('_'*45)


# In[16]:


# gender
data_2 = data.replace(
    {'gender' : {'Male' : 0, 'Female' : 1, 'Other' : 2}}
)

# ever_married
data_2 =  data_2.replace(
    {'ever_married' : {'Yes' : 0, 'No' : 1}}
)

# work_type
data_2 =  data_2.replace(
    {'work_type' : {'Private' : 0, 'Self-employed' : 1, 'Govt_job' : 2, 'children' : 3, 'Never_worked' : 4}}
)

# smoking_status
data_2 =  data_2.replace(
    {'smoking_status' : {'formerly smoked' : 0, 'never smoked' : 1, 'smokes' : 2, 'Unknown' : 3}}
)

# Residence_type
data_2 =  data_2.replace(
    {'Residence_type' : {'Urban' : 0, 'Rural' : 1}}
)


# In[17]:


data_2.head()


# ## Step 3.5 | Normalization

# #### ➡️ Define X & y

# In[18]:


X_temp = data_2.drop(columns='stroke')
y = data_2.stroke


# #### ➡️ To decrease effect of larg values, we use MinMaxScaler to normalize X.

# In[19]:


scaler = MinMaxScaler().fit_transform(X_temp)
X = pd.DataFrame(scaler, columns=X_temp.columns)
X.describe()


# # Step 4 | Modeling

# ## Step 4.1 | Initialization

# In[20]:


# define a function to ploting Confusion matrix
def plot_confusion_matrix(y_test, y_prediction):
    cm = metrics.confusion_matrix(y_test, y_prediction)
    ax = plt.subplot()
    ax = sns.heatmap(cm, annot=True, fmt='', cmap="Greens")
    ax.set_xlabel('Prediced labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Dont Had Stroke', 'Had Stroke'])
    ax.yaxis.set_ticklabels(['Dont Had Stroke', 'Had Stroke']) 
    plt.show()


# In[21]:


# Splite X, y to train & test dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)


# ## Step 4.2 | RandomForestClassifier

# In[22]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'n_estimators' : [50, 100, 250, 500],
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'max_features' : ['sqrt', 'log2']
}

rf = RandomForestClassifier(n_jobs=-1)
rf_cv = GridSearchCV(estimator=rf, cv=10, param_grid=parameters).fit(X_train, y_train)

print('Tuned hyper parameters : ', rf_cv.best_params_)
print('accuracy : ', rf_cv.best_score_)


# In[23]:


# calculate time befor run algorithm
t1 = datetime.now()
# Model :
rf = RandomForestClassifier(**rf_cv.best_params_).fit(X_train, y_train)
# calculate time after run algorithm
t2 = datetime.now()


# In[24]:


y_pred_rf = rf.predict(X_test)

rf_score = round(rf.score(X_test, y_test), 3)
print('RandomForestClassifier score : ', rf_score)


# In[25]:


delta = t2-t1
delta_rf = round(delta.total_seconds(), 3)
print('RandomForestClassifier takes : ', delta_rf, 'Seconds')


# In[26]:


plot_confusion_matrix(y_test, y_pred_rf)


# In[27]:


cr = metrics.classification_report(y_test, y_pred_rf)
print(cr)


# ## Step 4.3 | LogisticRegression

# In[28]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
    'class_weight' : ['balanced'],
    'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}

lr = LogisticRegression()
lr_cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=10).fit(X_train, y_train)

print('Tuned hyper parameters : ', lr_cv.best_params_)
print('accuracy : ', lr_cv.best_score_)


# In[29]:


# Calculate time befor run algorithm
t1 = datetime.now()
# Model
lr = LogisticRegression(**lr_cv.best_params_).fit(X_train, y_train)
# Calculate time after run algorithm
t2 = datetime.now()


# In[30]:


y_pred_lr = lr.predict(X_test)

lr_score = round(lr.score(X_test, y_test), 3)
print('LogisticRegression score : ', lr_score)


# In[31]:


delta = t2-t1
delta_lr = round(delta.total_seconds(), 3)
print('LogisticRegression takes : ', delta_lr, 'Seconds')


# In[32]:


plot_confusion_matrix(y_test, y_pred_lr)


# In[33]:


cr = metrics.classification_report(y_test, y_pred_lr)
print(cr)


# ## Step 4.4 | SVC

# In[34]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
    'gamma' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
}



svc = SVC()
svc_cv = GridSearchCV(estimator=svc, param_grid=parameters, cv=10).fit(X_train, y_train)



print('Tuned hyper parameters : ', svc_cv.best_params_)
print('accuracy : ', svc_cv.best_score_)


# In[35]:


# Calculate time befor run algorithm
t1 = datetime.now()
# Model
svc = SVC(**svc_cv.best_params_).fit(X_train, y_train)
# Calculate time after run algorithm
t2 = datetime.now()


# In[36]:


y_pred_svc = svc.predict(X_test)

svc_score = round(svc.score(X_test, y_test), 3)
print('SVC Score : ', svc_score)


# In[37]:


delta = t2-t1
delta_svc = round(delta.total_seconds(), 3)
print('SVC : ', delta_svc, 'Seconds')


# In[38]:


plot_confusion_matrix(y_test, y_pred_svc)


# In[39]:


cr = metrics.classification_report(y_test, y_pred_svc)
print(cr)


# ## Step 4.5 | DecisionTreeClassifier

# In[40]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'splitter' : ['best', 'random'],
    'max_depth' : list(np.arange(4, 30, 1))
        }



tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(estimator=tree, cv=10, param_grid=parameters).fit(X_train, y_train)



print('Tuned hyper parameters : ', tree_cv.best_params_)
print('accuracy : ', tree_cv.best_score_)


# In[41]:


# Calculate time befor run algorithm :
t1 = datetime.now()
# Model :
tree = DecisionTreeClassifier(**tree_cv.best_params_).fit(X_train, y_train)
# Calculate time after run algorithm :
t2 = datetime.now()


# In[42]:


y_pred_tree = tree.predict(X_test)

tree_score = round(tree.score(X_test, y_test), 3)
print('DecisionTreeClassifier Score : ', tree_score)


# In[43]:


delta = t2-t1
delta_tree = round(delta.total_seconds(), 3)
print('DecisionTreeClassifier takes : ', delta_tree, 'Seconds')


# In[44]:


plot_confusion_matrix(y_test, y_pred_tree)


# In[45]:


cr = metrics.classification_report(y_test, y_pred_tree)
print(cr)


# ## Step 4.6 | KNeighborsClassifier

# In[46]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'n_neighbors' : list(np.arange(3, 20, 2)),
    'p' : [1, 2, 3, 4]
}

# calculate time to run in second
t1 = datetime.now()

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(estimator=knn, cv=10, param_grid=parameters).fit(X_train, y_train)

t2 = datetime.now()

print('Tuned hyper parameters : ', knn_cv.best_params_)
print('accuracy : ', knn_cv.best_score_)


# In[47]:


# Calculate time befor run algorithm :
t1 = datetime.now()
# Model :
knn = KNeighborsClassifier(**knn_cv.best_params_).fit(X_train, y_train)
# Calculate time after run algorithm :
t2 = datetime.now()


# In[48]:


y_pred_knn = knn_cv.predict(X_test)

knn_score = round(knn.score(X_test, y_test), 3)
print('KNeighborsClassifier Score :', knn_score)


# In[49]:


delta = t2-t1
delta_knn = round(delta.total_seconds(), 3)
print('KNeighborsClassifier takes : ', delta_knn, 'Seconds')


# In[50]:


plot_confusion_matrix(y_test, y_pred_knn)


# In[51]:


cr = metrics.classification_report(y_test, y_pred_knn)
print(cr)


# ## Step 4.7 | Result

# In[52]:


result = pd.DataFrame({
    'Algorithm' : ['RandomForestClassifier', 'LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'KNeighborsClassifier'],
    'Score' : [rf_score, lr_score, svc_score, tree_score, knn_score],
    'Delta_t' : [delta_rf, delta_lr, delta_svc, delta_tree, delta_knn]
})

result


# In[53]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))

sns.barplot(x='Algorithm', y='Score', data=result, ax=ax[0])
ax[0].bar_label(ax[0].containers[0], fmt='%.3f')
ax[0].set_xticklabels(labels=result.Algorithm, rotation=300)

sns.barplot(x='Algorithm', y='Delta_t', data=result, ax=ax[1])
ax[1].bar_label(ax[1].containers[0], fmt='%.3f')
ax[1].set_xticklabels(labels=result.Algorithm, rotation=300)
plt.show()


# # Results :
# 

# - Acording to the above plots, best algorithms base on Score are :
#     - RandomForestClassifier
#     - SVC
#     - DecisionTreeClassifier
#     - KNeighborsClassifier
# 
# - And best Algorithm base on runtime, are :
#     - DecisionTreeClassifie
#     - KNeighborsClassifier
# 
# 
# - We choose  KNeighborsClassifier 

# ## Step 4.8 | Final Modeling 

# In[54]:


knn = KNeighborsClassifier(**knn_cv.best_params_).fit(X, y)
knn


# In[55]:


knn.score(X, y)


# In[ ]:




