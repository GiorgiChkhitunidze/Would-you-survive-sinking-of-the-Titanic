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


st.write("<h1 style='text-align:center'>Would you survive sinking of the Titanic?</h1>", 
         "<p style='text-align:center'><i>To find out your chances of survival answer the questions on the left sidebar.</i></p><br><br>",
         unsafe_allow_html=True)

st.sidebar.write('<h4><u>User Input</u></h4>', unsafe_allow_html=True)

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
            'Embarkation Port': Embarked}
    return pd.DataFrame(data, index=[0])

data = get_features()

st.write("<div style='margin-left: 10%;'><h4><u>User Input Parameters</u></h4></div>", unsafe_allow_html=True)
st.write(
    """
    <style>
        table {
            font-family: arial, sans-serif;
            border: 3px solid #07375E;
            border-collapse: collapse;
            width: 80%;
            margin-left: 10%;
            }
        
        th {
            background-color: #07375E;
        }

        td {
            background-color: #09406E;
            }
    </style>
    """ +
    f"""
    <table>
        <tr>
            <th>Sex</th>
            <th>Age</th>
            <th>№ of Siblings and/or Spouse</th>
            <th>№ of Parents and/or children</th>
            <th>Socio-econocmic status</th>
            <th>Fare</th>
            <th>Embarkation Port</th>
        </tr>
        <tr>
            <td>{data.iloc[0, 0]}</td>
            <td>{data.iloc[0, 1]}</td>
            <td>{data.iloc[0, 2]}</td>
            <td>{data.iloc[0, 3]}</td>
            <td>{data.iloc[0, 4]}</td>
            <td>{data.iloc[0, 5]}</td>
            <td>{data.iloc[0, 6]}</td>
        </tr>
    </table>
    """, 
    unsafe_allow_html=True
    )

# prepare user unput data for prediction
pclass_map = {"Upper Class" : 1, "Middle Class" : 2, "Lower Class" : 3}
data['Socio-econocmic status'] = data['Socio-econocmic status'].map(pclass_map)

sex_map = {'Female' : 0, 'Male' : 1}
data['Sex'] = data['Sex'].map(sex_map)

data['Embarked_Q'] = 0
data['Embarked_S'] = 0

if data.loc[0, 'Embarkation Port'] == 'Queenstown':
    data['Embarkation Port'] = 1
if data.loc[0, 'Embarkation Port'] == 'Southampton':
    data['Embarkation Port'] = 1

data.drop('Embarkation Port', axis=1, inplace=True)
data = data[[
    'Socio-econocmic status', 'Age', '№ of Siblings and/or Spouse',
    '№ of Parents and/or children', 'Fare', 'Sex', 'Embarked_Q',
    'Embarked_S'
    ]]
data.columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',
    'Embarked_S']

# load model and predict
st.write("<br><br><div style='margin-left: 10%;'><h4><u>Prediction</u></h4></div>", unsafe_allow_html=True)
model_Pickle = pickle.load(open('model_Pickle', 'rb'))
preds = model_Pickle.predict(data)
prediction_proba = model_Pickle.predict_proba(data)


# print results
if prediction_proba[0][0]>=0.6:
    st.write(f"<div style='margin-left: 10%;'><p>With <span style='color:red'>{int((prediction_proba[0][preds][0]*100).round())}%</span> confidence I can say you would <span style='color:red'>sink alongside the Titanic!</span>🥲</p></div>", unsafe_allow_html=True)
elif prediction_proba[0][0]>0.5:
    st.write("<div style='margin-left: 10%;'><p>Well, the results are ambiguous. Though, your chances are <span style='color:red'>slightly more skewed towards NOT surviving.</span>🤨</p></div>", unsafe_allow_html=True)
elif prediction_proba[0][0]==0.5:
    st.write("<div style='margin-left: 10%;'><p>Well, what do you know. Your chances are <span style='color:red'>fifty-fifty!</span>😏</p></div>", unsafe_allow_html=True)
elif prediction_proba[0][0]<0.4:
    st.write(f"<div style='margin-left: 10%;'><p>With <span style='color:red'>{int((prediction_proba[0][preds][0]*100).round())}%</span> confidence I can say you would <span style='color:red'>survive sinking of the Titanic!</span>😉👌</p></div>", unsafe_allow_html=True)
else:
    st.write("<div style='margin-left: 10%;'><p>Well, the results are ambiguous. Still, your chances are <span style='color:red'>slightly more skewed towards surviving.</span>😊</p></div>", unsafe_allow_html=True)