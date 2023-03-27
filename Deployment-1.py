#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pickle
import pandas as pd
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


# In[3]:

loaded_model = pickle.load(open('filename', 'rb'))


st.title('Electric Motor Temperature: Motor Speed Prediction')
st.write('Random Forest Regressor')


# In[4]:


st.sidebar.header('User Input Parameters')


# In[5]:


def user_input_features():
    Ambient = st.sidebar.number_input("Insert ambient")
    Coolant = st.sidebar.number_input("Insert Coolant value")
    u_d = st.sidebar.number_input("Insert u_d")
    u_q = st.sidebar.number_input("Insert u_q")
    Torque = st.sidebar.number_input("Insert Torque")
    i_d = st.sidebar.number_input("Insert i_d")
    i_q = st.sidebar.number_input("Insert i_q")
    pm = st.sidebar.number_input("Insert pm")
    stator_yoke = st.sidebar.number_input("Insert stator_yoke")
    stator_tooth = st.sidebar.number_input("Insert stator_tooth")
    stator_winding = st.sidebar.number_input("Insert stator_winding")
    data = {'ambient':Ambient,
            'coolant':Coolant,
            'u_d':u_d,
            'u_q':u_q,
            'torque':Torque,
            'i_d':i_d,
            'i_q':i_q,
            'pm':pm,
            'stator_yoke':stator_yoke,
            'stator_tooth':stator_tooth,
            'stator_winding':stator_winding}
    features = pd.DataFrame(data,index = [0])
    return features 


# In[6]:


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# In[7]:


motor = pd.read_csv("temperature_data.csv")
motor.drop(["profile_id"],inplace=True,axis = 1)
motor = motor.dropna()




# In[ ]:


st.subheader('Motor speed Prediction')


# In[ ]:

empty = ''
if st.button('Predict The Motor speed'):
        empty = RandomForest([ambient,coolant,u_d,u_q,i_d,i_q,pm,stator_winding,stator_tooth,stator_yoke,torque])
        
        st.success(empty)


