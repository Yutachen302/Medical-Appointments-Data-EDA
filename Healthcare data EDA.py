#!/usr/bin/env python
# coding: utf-8

# In[177]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[178]:


data = pd.read_csv("Data.csv")


# In[179]:


data.info()


# In[180]:


data.head()


# In[181]:


data.shape


# In[182]:


###No NA value


# # Convert ScheduledDay &AppiontmentDay to only date, remove the hours

# In[183]:


data["ScheduledDay"] = pd.to_datetime(data['ScheduledDay']).dt.date.astype('datetime64[ns]')
data["AppointmentDay"] = pd.to_datetime(data['AppointmentDay']).dt.date.astype('datetime64[ns]')


# In[184]:


data.head()


# In[ ]:





# # Create new column for day of the week (0=Monday, 1=Tuesday, etc)

# In[185]:


data['sch_weekday'] = data["ScheduledDay"].dt.dayofweek


# In[186]:


data["app_weekday"] = data["AppointmentDay"].dt.dayofweek


# In[187]:


data.head()


# In[188]:


data["sch_weekday"].hist(bins=10)
plt.xlabel("Schedule Weekday")
#Barchart to show the distribution
#Patients schedule most appointment on Tuesday


# In[189]:


data['sch_weekday'].value_counts()


# In[190]:


data["app_weekday"].hist(bins=10)
plt.xlabel("Appointment Weekday")

#Wednesday has the most patient appointment 


# In[191]:


data["app_weekday"].value_counts()


# In[ ]:





# # Correct the column name

# In[192]:


#check each column
data.columns


# In[193]:


#Fix the namne of columns
data=data.rename(columns={'Hipertension':'Hypertension', 'Handcap':'Handicap', 'No-show':'Noshow'})


# In[194]:


data.columns


# In[195]:


data.info()


# In[196]:


#Remove unnucssary columns 
data.drop(['PatientId','AppointmentID','Neighbourhood'], axis=1, inplace=True)


# In[197]:


data.head()


# In[198]:


data.describe()
#For numerical value only


# # Plotting target variable: No Show

# In[199]:


#use value count to convert Noshow data into numerical for plotting
data['Noshow'].value_counts().plot(kind='barh')
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)


# In[200]:


data['Noshow'].value_counts()


# In[201]:


#percentage of no show
data.shape


# In[202]:


no_show_rate= 22319/110527
print(no_show_rate)
#About 20% no show rate


# In[ ]:





# # Check for missing value (No missing value)

# In[203]:


missing = pd.DataFrame((data.isnull().sum())*100/data.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()

#No missing value


# In[ ]:





# # Data cleaning (convert age into groups)

# In[204]:


cat = pd.cut(data.Age,bins=[0,2,17,65,115],
                  labels=['Toddler/baby','Child','Adult','Elderly'])
data.insert(5,'Age Group',cat)


# In[205]:


data['Age Group'].value_counts()


# ### Toddler/baby: 0-2
# ### Child: 3-17
# ### Adult: 18-65
# ### Elderly: >65

# In[206]:


#Remove age column
data.drop(['Age'], axis=1, inplace=True)
list(data)


# # EDA

# ## Gender vs No show

# In[207]:


sns.countplot(x='Gender', data=data, hue='Noshow',palette='viridis')


# ## Schedule weekday vs No show

# In[208]:


sns.countplot(x='sch_weekday', data=data, hue='Noshow',palette='viridis')


# ## Appointment weekday vs No show

# In[209]:


sns.countplot(data=data, x='app_weekday', hue='Noshow', palette='viridis')


# # Age vs No show

# In[210]:


sns.countplot(data=data, x='Age Group', hue='Noshow', palette= 'viridis')


# # Scholarship vs Noshow

# In[211]:


sns.countplot(data=data, x='Scholarship', hue='Noshow', palette= 'viridis' )


# # Hypertension vs No show

# In[212]:


sns.countplot(data=data, x='Hypertension', hue='Noshow', palette= 'viridis' )


# # Diabetes vs No show 

# In[213]:


sns.countplot(data=data, x='Diabetes', hue='Noshow', palette= 'viridis' )


# # Alcoholism vs No show

# In[214]:


sns.countplot(data=data, x='Alcoholism', hue='Noshow', palette= 'viridis' )


# In[215]:


list(data)


# # Handicap vs No show

# In[216]:


sns.countplot(data=data, x='Handicap', hue='Noshow', palette='viridis')


# # SMS_received vs No show

# In[217]:


sns.countplot(data=data, x='SMS_received', hue='Noshow', palette='viridis')


# In[ ]:





# # Check correlation of each variable with No show

# ## Convert categorical variables to dummies

# In[218]:


data['Noshow'] = np.where(data.Noshow == 'Yes',1,0)
#Target variable: no show


# In[219]:


data['Noshow'].value_counts()


# In[220]:


dummies_data = pd.get_dummies(data)
dummies_data.head()


# In[221]:


#Correlation
plt.figure(figsize=(20,10))
dummies_data.corr()['Noshow'].sort_values(ascending = False).plot(kind='bar')


# ### SMS shows strong correation with no show, following with child and scholarship

# In[ ]:





# ## Visualize correlation with heatmap

# In[222]:


plt.figure(figsize=(11,11))
sns.heatmap(data.corr(), cmap="Paired")


# # Findings

# ##### 1. Female patients has taken more appointments than male patients
# ##### 2. Appointments scheduled on Tuesday have the highest no show rate
# ##### 3. Tuesday and Wednesday has the most no show, following by Monday, Thursday and Friday
# ##### 4. There are around 100000 patients without Scholarship and out of them around 80% have come for the visit and out of the 21000 patients with Scholarship around 75% of them have come for the visit.
# ##### 5. There are around 75000 patients who have not received SMS and out of them around 84% have come for the visit and out of the 35000 patients who have received SMS around 70% of them have come for the visit.

# In[ ]:




