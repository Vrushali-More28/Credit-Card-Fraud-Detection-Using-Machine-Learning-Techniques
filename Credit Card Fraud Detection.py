#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install scikit-plot -q')
get_ipython().system('pip install graphviz -q')
get_ipython().system('pip install pydotplus -q')


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scikitplot as skplt


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


from imblearn.under_sampling import RandomUnderSampler


# In[9]:


from sklearn.metrics import classification_report


# In[10]:


from sklearn.metrics import confusion_matrix


# In[11]:


from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


from sklearn.tree import DecisionTreeClassifier


# In[14]:


from IPython.display import Image  


# In[15]:


from sklearn.tree import export_graphviz


# In[16]:


import pydotplus


# In[17]:


import warnings


# In[18]:


warnings.filterwarnings('ignore')


# In[19]:


import matplotlib.pyplot as plt


# In[20]:


import seaborn as sns


# In[21]:


# additional configuration
plt.style.use('ggplot')
sns.set_style('dark')


# In[22]:


import pandas as pd


# In[23]:


# configure to display all lines and rows on the results
pd.options.display.max_rows = None
pd.options.display.max_columns = None


# In[24]:


# import credit card data set and save it in the 'df' variable
df = pd.read_csv('https://www.dropbox.com/s/b44o3t3ehmnx2b7/creditcard.csv?dl=1')


# In[25]:


# print the 5 first entries
df.head()


# In[26]:


# print the 5 last entries
df.tail()


# In[26]:


# check the data set size
print('Data set Size')
print('-' * 30)
print('Total de entries:\t {}'.format(df.shape[0]))
print('Total de attributes:\t {}'.format(df.shape[1]))


# In[28]:


# get data set information
df.info()


# In[27]:


# check the amount of missing data
df.isnull().sum()


# In[30]:


# total transactions and percentage of fraude
print('Legitimate transactions: {}'.format(df.Class.value_counts()[0]))
print('Fraud transactions: {}'.format(df.Class.value_counts()[1]))
print('-' * 30)
print('Fraud transactions represent {:.2f}% of the data set.'.format(((df.Class.value_counts()[1]) * 100) / df.shape[0]))


# In[28]:


# Print the shape of the data
# data = data.sample(frac=0.1, random_state = 48)
print(df.shape)
print(df.describe())

# V1 - V28 are the results of a PCA Dimensionality reduction to protect user identities and sensitive features


# In[29]:


# plot bar chart for legit vs fraud transactions
## define axes
x = ['Legit', 'Fraud']
y = df.Class.value_counts()

## set color configuration
bar_colors = ['#bdbdbd', '#004a8f']

## plot chat
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(x, y, color=bar_colors)

### title
ax.text(-0.5, 1000000, 'Total Transactions', fontsize=20, color='#004a8f', 
        fontweight='bold')

### subtitle
ax.text(-0.5, 450000, 'Total transactions made with credit cards\n'
        'in 2 days period in which it was identified\n'
        'the ones that were a fraud.', fontsize=8, color='#6d6e70')

### legit transactions
ax.text(-0.15, 170000, '284.315', fontsize=12, color="#6d6e70")

### fraudulent transactions
ax.text(0.9, 550, '492', fontsize=14, color="#004a8f", fontweight='bold')

### set edges
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

### set logarithmic scale
ax.set_yscale('log')

plt.show()


# In[30]:


# distribution of Time
time = df['Time'].values
sns.distplot(time)


# In[31]:


# filter legitimate transactions and generate summary statistics of 'Amount'
print('LEGIT'.center(20))
print(' ')
print('{}'.format(df.Amount.loc[df.Class == 0].describe().round(2)))
print(' ')
print('-' * 20)
print(' ')

# filter fraudulent transactions and generate summary statistics of 'Amount'
print('FRAUD'.center(20))
print(' ')
print('{}'.format(df.Amount.loc[df.Class == 1].describe().round(2)))


# In[32]:


# plot density charts
## create variable with PCA (V1, V2, V3, ..., V28) columns
pca_columns = df.drop(['Class', 'Amount', 'Time'], axis=1).columns

## target legitimate and fraudulent transactions
class_0 = df[df.Class == 0]
class_1 = df[df.Class == 1]

# set chart
fig, ax = plt.subplots(nrows=7, ncols=4, figsize=(18,18))
fig.subplots_adjust(hspace=1, wspace=1)

# loop to plot all 28 variables
idx = 0
for col in pca_columns:
    idx += 1
    plt.subplot(7, 4, idx)
    sns.kdeplot(class_0[col], shade=True, color='b')
    sns.kdeplot(class_1[col], shade=True, color='r')

plt.tight_layout()


# In[33]:


# Plot histograms of each parameter 
df.hist(figsize = (20, 20))
plt.show()


# In[34]:


Fraud = df[df['Class'] == 1]
Valid = df[df['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(df[df['Class'] == 1])))
print('Valid Transactions: {}'.format(len(df[df['Class'] == 0])))


# In[35]:


# Correlation matrix
corrmat = df.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[36]:


# make a dataset copy
df_clean = df.copy()


# In[37]:


# separate independent variables from the dependent variable
X = df_clean.drop('Class', axis=1)
y = df['Class']

# split training and test data
## stratify= y (to divide so that the classes have the same proportion)
## random_state so that the result is replicable
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y, shuffle=True,
                                                    random_state=110)


# In[38]:


from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix


# In[39]:


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)

print('Dimensions of sets BEFORE treatment' + '\n' + '-' * 45)
print('TRAINING')
print('X_train: {}'.format(X_train.shape))
print('y_train: {}'.format(y_train.shape))
print('-' * 18)
print('TEST')
print('X_test: {}'.format(X_test.shape))
print('y_test: {}'.format(y_test.shape))
print(' ')
print('\nDimensions of sets AFTER treatment' + '\n' + '-' * 45)
print('TRAINING')
print('X_rus: {}'.format(X_rus.shape))
print('y_rus: {}'.format(y_rus.shape))
print('-' * 18)
print('TEST')
print('X_test: {}'.format(X_test.shape))
print('y_test: {}'.format(y_test.shape))


# In[40]:


# calculate correlations
corr = X_train.corr()
corr_rus = X_rus.corr()


# In[52]:


# plot heatmap
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
fig.suptitle('Correlation Matrix', fontsize=16, y=1.1, color='black')

# unbalanced data
ax[0].set_title('Unbalanced Data', fontsize=12, color='black')
sns.heatmap(corr, cmap='bwr_r', cbar=False, square=True, xticklabels=True, 
            yticklabels=True, linewidths=.1, ax=ax[0])

# balanced data
ax[1].set_title('Balanced Data', fontsize=12, color='black')
sns.heatmap(corr_rus, cmap='bwr_r', cbar=True, square=True, xticklabels=True, 
            yticklabels=True, linewidths=.1, ax=ax[1])

plt.show()


# In[41]:


#Building another model/classifier ISOLATION FOREST
from sklearn.ensemble import IsolationForest
ifc=IsolationForest(max_samples=len(X_train),
                    contamination=outlier_fraction,random_state=1)
ifc.fit(X_train)
scores_pred = ifc.decision_function(X_train)
y_pred = ifc.predict(X_test)


# Reshape the prediction values to 0 for valid, 1 for fraud. 
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

n_errors = (y_pred != y_test).sum()


# In[44]:


#evaluation of the model
#printing every score of the classifier
#scoring in any thing

from sklearn.metrics import confusion_matrix
n_outliers = len(Fraud)
print("the Model used is {}".format("Isolation Forest"))
acc= accuracy_score(y_test,y_pred)
print("The accuracy is  {}".format(acc))
prec= precision_score(y_test,y_pred)
print("The precision is {}".format(prec))
rec= recall_score(y_test,y_pred)
print("The recall is {}".format(rec))
f1= f1_score(y_test,y_pred)
print("The F1-Score is {}".format(f1))
MCC=matthews_corrcoef(y_test,y_pred)
print("The Matthews correlation coefficient is{}".format(MCC))

#printing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, xticklabels=LABELS,
            yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Run classification metrics
plt.figure(figsize=(9, 7))
print('{}: {}'.format("Isolation Forest", n_errors))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[45]:


# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# predictions
y_pred = rfc.predict(X_test)


# In[47]:


#Evaluating the classifier
#printing every score of the classifier
#scoring in any thing
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix
n_outliers = len(Fraud)
n_errors = (y_pred != y_test).sum()
print("The model used is Random Forest classifier")
acc= accuracy_score(y_test,y_pred)
print("The accuracy is  {}".format(acc))
prec= precision_score(y_test,y_pred)
print("The precision is {}".format(prec))
rec= recall_score(y_test,y_pred)
print("The recall is {}".format(rec))
f1= f1_score(y_test,y_pred)
print("The F1-Score is {}".format(f1))
MCC=matthews_corrcoef(y_test,y_pred)
print("The Matthews correlation coefficient is {}".format(MCC))


#printing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Run classification metrics
plt.figure(figsize=(9, 7))
print('{}: {}'.format("Random Forest", n_errors))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[48]:


# Building the Decision Tree Classifier (DECISION TREE)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Building the Decision Tree Classifier
np.random.seed(29)
tree_model = DecisionTreeClassifier(max_depth=6, criterion='entropy')
tree_model.fit(X_train, y_train)
tree_yhat = tree_model.predict(X_test)


# In[49]:


#Evaluating the classifier
#printing every score of the classifier
#scoring in any thing

# Evaluating the classifier
print("The model used is Decision Tree Classifier")

acc = accuracy_score(y_test, tree_yhat)
print("The accuracy is  {}".format(acc))

prec = precision_score(y_test, tree_yhat)
print("The precision is {}".format(prec))

rec = recall_score(y_test, tree_yhat)
print("The recall is {}".format(rec))

f1 = f1_score(y_test, tree_yhat)
print("The F1-Score is {}".format(f1))

MCC = matthews_corrcoef(y_test, tree_yhat)
print("The Matthews correlation coefficient is {}".format(MCC))

# Printing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, tree_yhat)
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Run classification metrics
plt.figure(figsize=(9, 7))
print(classification_report(y_test, tree_yhat))


# In[ ]:




