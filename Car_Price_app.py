import streamlit as st
import pickle as pkl
import sklearn
import numpy as np

pipe = pkl.load(open('pipe.pkl', 'rb'))
df1 = pkl.load(open('df1.pkl', 'rb'))

st.title('Car price predictor')

# brand
company = st.selectbox('Brand', df1['Company'].unique())

# types of body
body = st.selectbox('Body', df1['carbody'].unique())

# fuel
fuel = st.selectbox('Fuel type', df1['fueltype'].unique())

# enginesize
enginesize = st.number_input('Engine size')
st.text('enter value between 60 to 300')

# horsepower
horsepower = st.number_input('horsepower')
st.text('enter value between 50 to 300')

# torque
torque = st.number_input('torque')
st.text('enter value between 50 to 250')

# citympg
mileage_in_city = st.number_input('city mileage')
st.text('enter value between 13 to 40')

# highway mpg
mileage_on_highway = st.number_input('highway mileage')
st.text('enter the value between 15 to 50')

# peak rpm
rpm = st.number_input('rpm')
st.text('enter the values from 4000 to 6000')

if st.button('Predict price'):
    query = np.array([company, body, fuel, enginesize, horsepower, torque, mileage_in_city, mileage_on_highway, rpm])
    query = query.reshape(1, 9)
    st.title(pipe.predict(query))


