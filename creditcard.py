#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[17]:


credit_data = pd.read_csv("creditcard.csv")
credit_data.head()


# In[18]:


credit_data.tail()


# In[19]:


credit_data.shape


# In[20]:


credit_data.info()


# In[21]:


credit_data.isnull().sum()


# In[22]:


credit_data.describe()


# In[23]:


credit_data.columns


# In[24]:


credit_data.Class.unique()


# In[25]:


fraud_case = credit_data[credit_data['Class']==1]
valid_case = credit_data[credit_data['Class']==0]

print(f"Number of Fraud Case: {len(fraud_case)}")
print(f"Number of Valid Case: {len(valid_case)}")

total = len(fraud_case)/len(valid_case)
total


# In[26]:


per = len(fraud_case)/len(credit_data['Class'])*100
per


# In[27]:


#count the number of occurance for each class(0 for legitimate,1 for fraudulent)
class_counts = credit_data['Class'].value_counts()
percentage_fraudulent =(class_counts[1]/class_counts.sum())*100
plt.figure(figsize=(8,6))
sns.countplot(x='Class', data=credit_data)
plt.title('Distribution of Legitimate vs. Fraudulent Transactions')
plt.xlabel('Class (0: Legitimate, 1: Fraudulent)')
plt.ylabel('Count')

plt.text(0, class_counts [0] + 1000, f'Percentage of Fraudulent Transactions: {percentage_fraudulent:2f}%',fontsize=12,ha='center')
plt.show


# In[28]:


plt.figure(figsize=(10, 6))
sns.histplot(credit_data['Time'], bins=48, kde=True) # Using 48 bins for a daily view
plt.title('Transaction Time Distribution')
plt.xlabel("Time (in seconds)")
plt.ylabel('Count')
#Check if there are specific times of day when fraud is more likely to occur
plt.figure(figsize=(10, 6))
sns.histplot(credit_data[credit_data['Class'] == 1] ['Time'], bins=48, kde=True, color='red', label='Fraudulent')
sns.histplot(credit_data[credit_data['Class'] == 0]['Time'], bins=48, kde=True, color='blue', label='Legitimate')
plt.title('Fraud vs. Legitimate Transaction Time Distribution')
plt.xlabel('Time (in seconds)')
plt.ylabel('Count')
plt.legend()
           
plt.show()


# In[29]:


# Separate legitimate and fraudulent transactions
legitimate_transactions = credit_data[credit_data['Class'] == 0]
fraudulent_transactions = credit_data[credit_data['Class'] == 1]
#Calculate summary statistics for both groups
legitimate_summary = legitimate_transactions.describe() 
fraudulent_summary = fraudulent_transactions.describe()
print("Summary Statistics for Legitimate Transactions:")
print(legitimate_summary)
print("\nSummary Statistics for Fraudulent Transactions:")
print(fraudulent_summary)


# In[30]:


#Calculate the correlation matrix
correlation_matrix = credit_data.corr()
print(correlation_matrix)
#Filter the correlations of features with the target variable ('Class')
feature_correlations= correlation_matrix['Class'].drop('Class')
print(feature_correlations)
correlation_threshold = 0.1
highly_correlated_features = feature_correlations [abs (feature_correlations) > correlation_threshold]
plt.figure(figsize=(12, 6))
sns.barplot(x=highly_correlated_features.index, y=highly_correlated_features.values, palette='viridis')
plt.title('Feature Correlations with Fraud (Class)')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.xticks (rotation=45)
plt.show


# In[31]:


plt.figure(figsize=(10, 6))
plt.scatter(credit_data['Time'], credit_data['Amount'], c=credit_data['Class'], cmap='coolwarm', alpha=0.5)
plt.title('Transaction Amount vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Transaction Amount')
plt.colorbar(label='Class (8: Legitimate, 1: Fraudulent)')
plt.show()


# In[32]:


correlation_matrix = credit_data.corr()
#Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap Between Features')
plt.show()


# In[33]:


#import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)


# In[34]:


#Test value
X = credit_data.drop(['Class'], axis=1)
# Target value
Y = credit_data['Class']
print(f"value and shapes {X.values, X.shape}")
print(f"value and shapes {Y.values, Y.shape}")


# In[35]:


X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)


# In[ ]:


model = RandomForestClassifier(n_estimators=100, random_state=42)

#Train the model on the training data
model.fit(X_train, Y_train)

#Make predictions on the test data 
y_pred= model.predict(X_test)


# In[ ]:


print(y_pred)
print(Y_test)


# In[ ]:


print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("\nClassification Report:\n", classification_report (Y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(Y_test, y_pred))


# In[ ]:




