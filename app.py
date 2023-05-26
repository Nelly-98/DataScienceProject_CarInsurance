import numpy as np 
import pandas as pd 
import joblib 
import streamlit as st
from PIL import Image
import pickle5 as pickle
import sklearn
from sklearn.preprocessing  import LabelEncoder
#from deep_translator import GoogleTranslator

st.set_page_config(
    page_title="Car Insurance ",
    page_icon="🧊",
    initial_sidebar_state="collapsed",
    #layout="wide"
)

df=pd.read_csv('insurance.csv')
df.drop(columns=['Unnamed: 0','ID','PrimeAssurance','ProduitAssurance'],inplace=True)

# WEB APP  


st.title("Quel est votre prime et la Meilleure offre pour vous ?")
#st.subheader("Voyons-voir comment l'Intelligence Artificielle nous permet de répondre.")

image = Image.open('Car-Insurance.jpg')
st.image(image,width=600)



# Deployement
#model = joblib.load(filename="final_model.joblib")

####### Features
with st.sidebar:

    ages=df['AGE'].unique()
    st.header("1. Choissisez la tranche d'age .")
    age = st.selectbox('',ages)
    ###

    genders=df['GENDER'].unique()
    st.header("2 . Selectionner le sexe")
    gender = st.selectbox("",genders)
    ###

    races=df['RACE'].unique()
    st.header("3 . Selectionner le genre ")
    race = st.radio("",races)
    ###

    exps=df['DRIVING_EXPERIENCE'].unique()
    st.header("4 . Selectionner votre experience de conduite")
    exp = st.selectbox("",exps)
    ###

    eds=df['EDUCATION'].unique()
    st.header("5 . Selectionner le niveau d'education")
    ed = st.selectbox("",eds)
    ###

    incomes=df['INCOME'].unique()
    st.header("6 . Selectionner la tranche de revenus")
    income = st.selectbox("",incomes)
    ###

    cr= st.number_input('7.Selectionner le credit score ')
    ###

    st.header("8. Etes vous le proprietaire du véhicule ")
    ownership = float(st.slider("",0,1))
    ###

    years=df['VEHICLE_YEAR'].unique()
    st.header("9 . Selectionner l'année du vehicule")
    year = st.selectbox("",years,key=70)
    ###

    genders=df['GENDER'].unique()
    st.header("10 . Etes vous marié ?")
    married = float(st.slider("",0,1,key=3))
    ###

    st.header("11 . Avez vous des enfants ?")
    kids = float(st.slider("",0,1,key=4))
    ###

    postal_codes=df['POSTAL_CODE'].unique()
    st.header("12 . Entrez votre code postal  ?")
    postal_code = int(st.selectbox("",postal_codes))
    ###

    st.header("13 .Quel est votre mileage annuel ?")
    mileage = float(st.number_input(""))
    ###
    st.header("14 . Quel est votre type de vehicles ?")
    vehicles=df['VEHICLE_TYPE'].unique()
    vehicle = st.radio("",vehicles,key=90)
    ###
    st.header("15 . Nombres d'infractions de vitesses ?")   
    violation = int(st.number_input("",key=15))
    ###
    st.header("16 . DIUS ?")
    dius =int(st.number_input("",key=20))
    ###
    st.header("17 Nombres d'accidents passés ?")
    past_accident = int(st.number_input("",key=30))
    ###
    st.header("18 .OUTCOME ?")
    outcome = float(st.slider("",0,1,key=11))
###
####################

inputs=[age,gender,race,exp,ed,income,cr,\
                    ownership,year,married,kids,postal_code,mileage,\
                        vehicle,violation,dius,past_accident,outcome]
#st.write(inputs)
st.subheader("  Récapitulatif ")
input_df=pd.DataFrame([inputs],columns=df.columns)
st.write("Nous allons maintenant chercher la meilleure offre pour vous à partir de vos  informations contenues dans le tableau suivant : ")
st.write(input_df.head())
# Transformations
#cat_cols=input_df.select_dtypes(include='object').columns
model = pickle.load(open('kmeans-model', 'rb'))
st.subheader(" C'est parti pour découvrir la formule la plus adaptée")

def calcul_prime(df):
    prime = np.where((df["MARRIED"] == 1) & (df["PAST_ACCIDENTS"] == 0) & (df["DUIS"] == 0), 1000,
                                np.where((df["MARRIED"] == 0.0) & (df["PAST_ACCIDENTS"] == 0) & (df["DUIS"] == 0), 1500,
                                         np.where((df["MARRIED"] == 0.0) & (df["PAST_ACCIDENTS"] > 0), 2000,
                                                  np.where((df["MARRIED"] == 0.0) & (df["DUIS"] > 0), 2500, 3000))))
    return prime
# Prediction 
prediction=1  
ms0=f"Assurance responsabilité civile :Prime {int(calcul_prime(input_df))}£"
ms1=f"Assurance  Standard : Prime  {int(calcul_prime(input_df))}£"
ms2=f"Assurance Tout risques : {int(calcul_prime(input_df))}£"
predict =st.button(':green[Predict]')
st.write(":point_up_2: Cliquez pour prédire.")
prediction=model.predict(input_df)
if predict:
    if prediction == 0 :
        st.success(ms0)
    elif prediction==1: 
        st.success(ms1)
    else :
        st.success(ms2)



