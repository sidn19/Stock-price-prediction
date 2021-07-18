# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 17:20:22 2021

@author: NiruSai
"""

import streamlit as st
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import inspect
import pickle





model_type=st.sidebar.selectbox(label='Select the Model', options=['Select the Model','Linear Regression' ,'Neural Network'])
if model_type=='Linear Regression':
    pickle_in=open("classifier.pkl","rb")
    model=pickle.load(pickle_in)
    def welcome():
        return "Welcome All"
    def predict_stock(Volume, Open, High, Low):
        prediction=model.predict([[Volume, Open, High, Low]])
        print(prediction)
        return prediction
    def main():
       st.title("APPLE-STOCK PREDICTION USING LINEAR REGRESSION ")
       html_temp = """
       <div style="background-color:teal;padding:10px">
       <h2 style="color:white;text-align:center;">Stock Prediction Using Streamlit </h2>
       </div>
       """
       st.markdown(html_temp,unsafe_allow_html=True)
       Volume = st.text_input("Volume")
       Open = st.text_input("Open")
       High = st.text_input("High")
       Low = st.text_input("Low")
       result=""
       if st.button("Predict"):
           result=predict_stock(Volume, Open, High, Low)
       st.success('The output is {}'.format(result))
    if __name__=='__main__':
        main()
elif model_type=='Neural Network':
    stock_data = pd.read_csv(
    "Stock_Data_with_vader.csv", parse_dates=["Date"], index_col="Date")
    st.title("APPLE-STOCK PREDICTION USING NEURAL NETWORK ")
    x_input = stock_data['Close/Last']
    days = len(x_input)-20 
    x_input = x_input[days:]
    scaler = MinMaxScaler(feature_range=(0,1))
    x_input = scaler.fit_transform(np.array(x_input).reshape(-1,1))
    x_input=x_input.reshape(1,-1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    
    today = stock_data.index[-1].date()
    end_date = st.date_input('Enter the date', datetime.date.today())
    if today < end_date:
        st.success('Start date: `%s`\n\nEnd date:`%s`' % (today, end_date))
        if st.button("Predict"):
            lst_output=[]
            n_steps=20
            num_days=end_date-today
            i=0
            model = keras.models.load_model("20daysprediction.h5")
            while(i<num_days.days):
                if(len(temp_input)>20):
                    x_input=np.array(temp_input[1:])
                    x_input=x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = model.predict(x_input,verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    temp_input=temp_input[1:]
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = x_input.reshape((1, n_steps,1))
                    yhat = model.predict(x_input,verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    lst_output.extend(yhat.tolist())
                    i=i+1
            lst_output=scaler.inverse_transform(lst_output)
            lst_output=np.around(lst_output,decimals=4)
            st.write("Prediction of close price for the given date is",lst_output[num_days.days-1][0])
            diff=lst_output[num_days.days-1][0] - lst_output[num_days.days-2][0]
            if(diff < 0):
                st.write("percentage decrease = ",round(((- (diff)/lst_output[num_days.days-2][0])*100),4))
            else:
                st.write("percentage increase = ",round((( (diff)/lst_output[num_days.days-2][0])*100),4))
            #st.line_chart(lst_output)
    else:
        st.error('Error: End date must fall after start date.')
else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:black';'text-align: center; color: black;'>APPLE STOCK-PREDICTION USING LINEAR REGRESSION and NEURAL NETWORK</h1>", unsafe_allow_html=True)
  
    

