import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pickle

# set up page name, icon, layout and sidebar behaviour
st.set_page_config(
    page_title="Titanic App",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# hide footer and main menu icon
hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


st.write("<h1 style='width:100%;margin:auto;'>Would you survive sinking of the Titanic?</h1>", 
         '<p><i>To find out your chances of survival answer the questions on the left sidebar.</i></p>', 
         unsafe_allow_html=True)

st.sidebar.header('User Input:')

def get_features():
    Sex = st.sidebar.radio(
        "What's your Gender?",
        ("Male", "Female")
    )
    Age = st.sidebar.slider(
        "What's your age?", 1, 80, 25, 1
        )
    SibSp = st.sidebar.slider(
        "Would you travel with siblings and/or spouse and with how many of them?", 0, 5, 1, 1
        )
    Parch = st.sidebar.slider(
        "Would you travel with parents and/or children and with how many of them?", 0, 6, 2, 1
        )
    Pclass = st.sidebar.radio(
        "What's your socio-economic status?",
        ("Upper Class", "Middle Class", "Lower Class")
        )
    Fare = st.sidebar.slider(
        "How much would you pay for the ticket? (0 if you won it or sneaked in)", 0, 100, 20, 1
        )
    Embarked = st.sidebar.radio(
        "There wera 3 embarkation ports. From which port would you board the ship?",
        ("Cherbourg", "Queenstown", "Southampton")
        )

    data = {'Sex': Sex,
            'Age': Age,
            '№ of Siblings and/or Spouse': SibSp,
            '№ of Parents and/or children': Parch,
            'Socio-econocmic status': Pclass,
            'Fare': Fare,
            'Embarked': Embarked}
    return pd.DataFrame(data, index=[0])

data = get_features()

st.subheader('User Input parameters')
st.table(data)

# prepare data for prediction
pclass_map = {"Upper Class" : 1, "Middle Class" : 2, "Lower Class" : 3}
data['Socio-econocmic status'] = data['Socio-econocmic status'].map(pclass_map)

sex_map = {'Female' : 0, 'Male' : 1}
data['Sex'] = data['Sex'].map(sex_map)

data['Embarked_Q'] = 0
data['Embarked_S'] = 0

if data.loc[0, 'Embarked'] == 'Queenstown':
    data['Embarked_Q'] = 1
if data.loc[0, 'Embarked'] == 'Southampton':
    data['Embarked_S'] = 1

data.drop('Embarked', axis=1, inplace=True)
data = data[[
    'Socio-econocmic status', 'Age', '№ of Siblings and/or Spouse',
    '№ of Parents and/or children', 'Fare', 'Sex', 'Embarked_Q',
    'Embarked_S'
    ]]
data.columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',
    'Embarked_S']

# load model and predict
st.subheader('Prediction')
model_Pickle = pickle.load(open('model_Pickle', 'rb'))
preds = model_Pickle.predict(data)
prediction_proba = model_Pickle.predict_proba(data)

# print results
if prediction_proba[0][0]>=0.6:
    st.write(f'<p>With <b style="color:red">{int((prediction_proba[0][preds][0]*100).round())}%</b> confidence I can say you would <b style="color:red">sink alongside the Titanic!</b>🥲</p>', unsafe_allow_html=True)
elif prediction_proba[0][0]>0.5:
    st.write('<p>Well, the results are ambiguous. Though, your chances are <b style="color:red">slightly more skewed towards NOT surviving.</b>🤨</p>', unsafe_allow_html=True)
elif prediction_proba[0][0]==0.5:
    st.write('<p>Well, what do you know. Your chances are <b style="color:red">fifty-fifty!</b>😏</p>', unsafe_allow_html=True)
elif prediction_proba[0][0]<0.4:
    st.write(f'<p>With <b style="color:red">{int((prediction_proba[0][preds][0]*100).round())}%</b> confidence I can say you would <b style="color:red">survive sinking of the Titanic!</b>😉👌</p>', unsafe_allow_html=True)
else:
    st.write('<p>Well, the results are ambiguous. Still, your chances are <b style="color:red">slightly more skewed towards surviving.</b>😊</p>', unsafe_allow_html=True)

