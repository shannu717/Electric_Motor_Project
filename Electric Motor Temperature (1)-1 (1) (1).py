#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


motor=pd.read_csv("temperature_data.csv")
motor.head()


# # EDA

# In[4]:


motor.shape


# In[5]:


motor.info()


# In[6]:


motor.describe().T


# In[7]:


#checking correlation with respect to torque
motor.corr()['torque']


# In[8]:


#checking correlation with respect to motor_speed
motor.corr()['motor_speed']


# In[9]:


corr_motor_speed=motor.corr()['motor_speed']


# In[10]:


y=corr_motor_speed.to_list()
x=motor.columns


# In[11]:


plt.figure(figsize=[11,8])
plt.scatter(x,y,)
plt.title("Correlation between Motor Speed and other features")
plt.xlabel("features", size=15)
plt.ylabel("correlation", size=15)
plt.xticks(x, [str(i) for i in x], rotation=90)

#set parameters for tick labels
plt.tick_params(axis='x', which='major', labelsize=10)

plt.tight_layout()
plt.show()


# In[11]:


#correlation map
cax=plt.subplots(figsize=(30,20))
corr=motor.corr()
sns.heatmap(corr, annot=True , linewidths=1, fmt='.2f', 
            mask= np.zeros_like(corr,dtype=np.bool),            
            cmap=sns.diverging_palette(100,200,as_cmap=True),
            square=True)

plt.show()


# In[12]:


ax=plt.subplots(figsize=(30,20))
#sns.heatmap(motor.corr())
sns.heatmap(motor.corr(), annot=True, linewidths=1, fmt='.2f', mask= np.zeros_like(motor.corr(),dtype=np.bool))
sns.heatmap(motor.corr(), annot=True, linewidths=1, fmt='.2f', mask= np.zeros_like(motor.corr(),dtype=np.bool))
            


# In[13]:


#checking for missing data 
motor.isnull().sum()


# In[14]:


motor.hist(figsize = (35,25))
plt.show()


# In[15]:


sns.pairplot(motor)


# In[16]:


#checking for duplicates in the dataset
motor1=motor.drop_duplicates(keep='first')


# In[17]:


motor1.shape


# There is no duplicate values present in the dataset

# In[18]:


sns.set(style="whitegrid", font_scale=1.8)
plt.subplots(figsize = (25,8))
sns.countplot('profile_id',data=motor).set_title('count of profile_id')


# In[19]:


#barplot for profile_id 
sns.set(style="whitegrid", font_scale=1.8)
plt.subplots(figsize = (25,8))
grpd = motor.groupby(['profile_id'])
_df = grpd.size().sort_values().rename('samples').reset_index()
ordered_ids = _df.profile_id.values.tolist()
sns.barplot(y='samples', x='profile_id', data=_df, order=ordered_ids).set_title('Count of profile_id')


# In[20]:


sns.set(style="whitegrid", font_scale=1.2)
plt.subplots(figsize = (55,25))
plt.subplot(3,3,1)
sns.boxplot(x='profile_id', y='motor_speed', data=motor)


# In[12]:


x=motor.drop("motor_speed",axis=1)
y=motor.iloc[:,5:6]
x.head()


# In[13]:


y.head()


# In[14]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)


# In[15]:


#shape of train and test
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)


# In[25]:


get_ipython().system('pip install sweetviz')
get_ipython().system('pip install pandas_profiling')
#pip install markupsafe
import sweetviz as sv 
sweet_report = sv.analyze(motor)
sweet_report.show_html('temperature_data.html')  


# In[26]:


import numpy as np
np.seterr(all='warn')
A = np.array([10])
a=A[-1]
a**a
import pandas_profiling as pp
EDA_report= pp.ProfileReport(motor)
EDA_report.to_file(output_file='report.html') 
EDA_report   


# In[16]:


motor.head()


# In[17]:


motor=motor.drop("profile_id",axis=1)


# In[18]:


motor.head()


# In[19]:


motor.corr()['ambient']
#ambient is NOT so highly corelated between other features also not corelated with the target variable. 


# In[20]:


motor.corr()['coolant']
#coolant is highly corelated with stator_yoke and stator_tooth and not corelated with target variable


# In[21]:


motor.corr()['u_q']
#u_q is NOT corelated with other features but HIGHLY corelated with target variable


# In[22]:


motor.corr()['u_d']
#u_d is highly corelated with Torque and I_q and not correlated with the target variable.


# In[23]:


motor.corr()['torque']
#torque is highly corelated with u_d and i_q and not corelated with target variable.


# In[24]:


motor.corr()['i_d']
#i_d is little bit corelated with stator_winding and HIGHLY corelated with target variable.


# In[25]:


motor.corr()['i_q']
#i_q is highly corelated with u_d and torque and not at all corelated with target variable.


# In[26]:


motor.corr()['pm']
#pm is corelated with ambient, stator_yoke, stator_tooth, stator_winding and nor corelated with target variable


# In[27]:


motor.corr()['stator_yoke']
#stator_yoke is highly corelated with coolant, pm, stator_tooth, stator_winding and not corelated with the target variable.


# In[28]:


motor.corr()['stator_tooth']
#stator_tooth is highly corelated with pm, coolant, stator_yoke, stator_winding and not corelated with target variable.


# In[29]:


motor.corr()['stator_winding']
#stator_winding is highly corelated with pm, stator_yoke , stator_tooth and not so corelated with target variable


# In[30]:


motor.corr()['motor_speed']


# In[31]:


from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf


# In[48]:


model=smf.ols('motor_speed~ambient+coolant+u_d+u_q+torque+i_q+i_d+pm+stator_yoke+stator_tooth+stator_winding',data=motor).fit()


# In[49]:


model.rsquared
#with out feature reduction we have got 92 percent acurracy which is really good.


# In[50]:


model.summary()


# In[60]:


#building the model by excluding on of the feature which are dependent to each other
model2=smf.ols('motor_speed~u_q+i_d+stator_winding+pm+stator_tooth+torque',data=motor).fit()


# In[61]:


model2.rsquared


# In[38]:


motor_cpy=motor.copy()


# In[39]:


motor_cpy['u_q_sqrd']=motor_cpy['u_q']**2
motor_cpy['i_d_sqrd']=motor_cpy['i_q']**2
motor_cpy['stator_tooth_sqrd']=motor_cpy['stator_tooth']**2
motor_cpy['stator_winding_sqrd']=motor_cpy['stator_winding']**2


# In[42]:


motor_cpy.head()
motor_cpy=motor_cpy.drop(['stator_tooth_sqrd','stator_winding_sqrd'],axis=1)


# In[35]:


model3=smf.ols('motor_speed~ambient+coolant+u_d+u_q+torque+i_q+i_d+pm+stator_yoke+stator_tooth+stator_winding+u_q_sqrd+i_d_sqrd',data=motor_cpy).fit()


# In[100]:


model3.rsquared


# In[101]:


model3.summary()
#after feature transformation the Rsquared value as exceptionaly increased.


# In[43]:


X=motor_cpy.drop(['motor_speed'],axis=1)


# In[44]:


Y=motor_cpy.iloc[:,4]


# In[45]:


Y.head()


# In[46]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)


# In[47]:


X.head()


# In[85]:


#Scaling the data


# In[48]:


from sklearn.preprocessing import StandardScaler 
from sklearn.compose import ColumnTransformer 

t=[('num',StandardScaler(),['ambient', 'coolant', 'u_d', 'u_q','torque', 'i_d', 'i_q', 'pm', 'stator_yoke','stator_tooth','stator_winding','u_q_sqrd','i_d_sqrd'])]
transformer=ColumnTransformer(transformers=t,remainder='passthrough') 
transformer.fit(X_train)  

# transform training data.
X_train = transformer.transform(X_train) 

# transform the test data.
X_test = transformer.transform(X_test) 


# In[49]:


from sklearn.linear_model import LinearRegression, ElasticNet 
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score 
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor


# In[57]:


#implementing DecisionTreeRegressor
DTR=DecisionTreeRegressor()
DTR.fit(X_train,Y_train)

#predicting using the test set
y_pred_DTR = DTR.predict(X_test)
MAE1=mean_absolute_error(Y_test,y_pred_DTR)
MSE1=mean_squared_error(Y_test,y_pred_DTR)
R2S1=r2_score(Y_test,y_pred_DTR)


# In[58]:


R2S1


# In[78]:


#implementing RandomForestRegressor
RFR=RandomForestRegressor()
RFR.fit(X_train,Y_train)

#predicting using the test set
y_pred_RFR = RFR.predict(X_test)
MAE2=mean_absolute_error(Y_test,y_pred_RFR)
MSE2=mean_squared_error(Y_test,y_pred_RFR)
R2S2=r2_score(Y_test,y_pred_RFR)


# In[79]:


R2S2


# In[60]:


#implementing ExtraTreeRegressor
ETR=ExtraTreeRegressor()
ETR.fit(X_train,Y_train)

#predicting using the test set
y_pred_ETR = ETR.predict(X_test)
MAE3=mean_absolute_error(Y_test,y_pred_ETR)
MSE3=mean_squared_error(Y_test,y_pred_ETR)
R2S3=r2_score(Y_test,y_pred_ETR)


# In[61]:


R2S3


# In[64]:


#implementing BaggingRegressor
BAG=BaggingRegressor()
BAG.fit(X_train,Y_train)

#predicting using the test set
y_pred_BAG = BAG.predict(X_test)
MAE4=mean_absolute_error(Y_test,y_pred_BAG)
MSE4=mean_squared_error(Y_test,y_pred_BAG)
R2S4=r2_score(Y_test,y_pred_BAG)


# In[65]:


R2S4


# In[66]:


#implementing AdaBoostRegressor
ADA=AdaBoostRegressor()
ADA.fit(X_train,Y_train)

#predicting using the test set
y_pred_ADA = ADA.predict(X_test)
MAE5=mean_absolute_error(Y_test,y_pred_ADA)
MSE5=mean_squared_error(Y_test,y_pred_ADA)
R2S5=r2_score(Y_test,y_pred_ADA)


# In[67]:


R2S5


# In[70]:


#implementing GradientBoostRegressor
GBR=GradientBoostingRegressor()
GBR.fit(X_train,Y_train)

#predicting using the test set
y_pred_GBR = GBR.predict(X_test)
MAE6=mean_absolute_error(Y_test,y_pred_GBR)
MSE6=mean_squared_error(Y_test,y_pred_GBR)
R2S6=r2_score(Y_test,y_pred_GBR)


# In[71]:


R2S6


# In[73]:


#implementing LinearRegressor
LR=LinearRegression()
LR.fit(X_train,Y_train)

#predicting using the test set
y_pred_LR = LR.predict(X_test)
MAE7=mean_absolute_error(Y_test,y_pred_LR)
MSE7=mean_squared_error(Y_test,y_pred_LR)
R2S7=r2_score(Y_test,y_pred_LR)


# In[74]:


R2S7


# In[76]:


#implementing Elastic net
ELN=ElasticNet()
ELN.fit(X_train,Y_train)

#predicting using the test set
y_pred_ELN = ELN.predict(X_test)
MAE8=mean_absolute_error(Y_test,y_pred_ELN)
MSE8=mean_squared_error(Y_test,y_pred_ELN)
R2S8=r2_score(Y_test,y_pred_ELN)


# In[77]:


R2S8


# In[83]:


#Result table
table={'Mean absolute error':[MAE1,MAE2,MAE3,MAE4,MAE5,MAE6,MAE7,MAE8],'Mean Square error':[MSE1,MSE2,
                                                MSE3,MSE4,MSE5,MSE6,MSE7,MSE8],'R_2 Score':[R2S1,R2S2,R2S3,
                                                                            R2S4,R2S5,R2S6,R2S7,R2S8]}
Index=['DecisionTreeRegressor','RandomForestRegressor','ExtraTreeRegressor','BaggingRegressor','AdaBoostRegressor',
         'GradientBoostRegressor','LinearRegressor','Elastic net']
result=pd.DataFrame(table,index=Index)


# In[84]:


result


# # RANDOM FOREST IS THE BEST FITTING MODEL WITH R2 SCORE 99.98%

# In[ ]:




