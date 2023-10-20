

# from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
# import flasgger
# from flasgger import Swagger
import streamlit as st

# app=Flask(__name__)
# Swagger(app)

pickle_in = open("logistic_regression_model.pkl","rb")
classifier=pickle.load(pickle_in)

# @app.route('/')
def welcome():
    return "Welcome All"

# @app.route('/predict',methods=["Get"])
def predict_note_authentication(OTI,	WTI,	ATI,	OLI,	OTI_A,	OTI_T, VL1,	VL2,	VL3,	IL1,	IL2,	IL3,	VL12,	VL23,	VL31,	INUT):

    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """
    # variance=request.args.get("variance")
    # skewness=request.args.get("skewness")
    # curtosis=request.args.get("curtosis")
    # entropy=request.args.get("entropy")
    OTI = float(OTI)
    WTI = float(WTI)
    ATI = float(ATI)
    OLI = float(OLI)
    OTI_A = float(OTI_A)
    OTI_T = float(OTI_T)
    VL1 = float(VL1)
    VL2 = float(VL2)
    VL3 = float(VL3)
    IL1 = float(IL1)
    IL2 = float(IL2)
    IL3 = float(IL3)
    VL12 = float(VL12)
    VL23 = float(VL23)
    VL31 = float(VL31)
    INUT = float(INUT)

    prediction=classifier.predict([[OTI,	WTI,	ATI,	OLI,	OTI_A,	OTI_T, VL1,	VL2,	VL3,	IL1,	IL2,	IL3,	VL12,	VL23,	VL31,	INUT]])
    prediction = prediction[0]
    answer = ""
    # print(prediction[0])
    if prediction == 1.0:
        answer = "Faulty!"
    elif prediction == 0.0:
        answer = "Everything looks good!"
    else:
        answer = "Unknown result"

    return "Status:-"+answer

# @app.route('/predict_file',methods=["POST"])
def main():
    st.title("Transformer Fault Prediction using Machine Learning")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Input Parameters for the Transformer</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    OTI = st.text_input("OTI", "Type Here")
    WTI = st.text_input("WTI", "Type Here")
    ATI = st.text_input("ATI", "Type Here")
    OLI = st.text_input("OLI", "Type Here")
    OTI_A = st.text_input("OTI_A", "Type Here")
    OTI_T = st.text_input("OTI_T", "Type Here")
    VL1 = st.text_input("VL1", "Type Here")
    VL2 = st.text_input("VL2", "Type Here")
    VL3 = st.text_input("VL3", "Type Here")
    IL1 = st.text_input("IL1", "Type Here")
    IL2 = st.text_input("IL2", "Type Here")
    IL3 = st.text_input("IL3", "Type Here")
    VL12 = st.text_input("VL12", "Type Here")
    VL23 = st.text_input("VL23", "Type Here")
    VL31 = st.text_input("VL31", "Type Here")
    INUT = st.text_input("INUT", "Type Here")
    prediction=""
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(OTI,	WTI,	ATI,	OLI,	OTI_A,	OTI_T, VL1,	VL2,	VL3,	IL1,	IL2,	IL3,	VL12,	VL23,	VL31,	INUT)


    st.success(f'The output is {result}')
    # st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()

