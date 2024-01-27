import numpy as np
import pickle
import pandas as pd
import streamlit as st

pickle_in1 = open("logistic_regression_model.pkl","rb")
lr_classifier=pickle.load(pickle_in1)

pickle_in2 = open("svc_model.pkl", "rb")
svc_classifier = pickle.load(pickle_in2)

pickle_in3 = open("knn_model.pkl", "rb")
knn_classifier = pickle.load(pickle_in3)

pickle_in3 = open("decision_model.pkl", "rb")
decision = pickle.load(pickle_in3)

pickle_in3 = open("random_forest_model.pkl", "rb")
random_forest = pickle.load(pickle_in3)

pickle_in3 = open("extra_trees_model.pkl", "rb")
et = pickle.load(pickle_in3)

pickle_in3 = open("adaboost_model.pkl", "rb")
ada = pickle.load(pickle_in3)

pickle_in3 = open("xgboost_model.pkl", "rb")
xgb = pickle.load(pickle_in3)



def predict_probability(OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23,	VL31, INUT):

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

    lr_pred = lr_classifier.predict([[OTI,	WTI,	ATI,	OLI,	OTI_A,	OTI_T, VL1,	VL2,	VL3,	IL1,	IL2,	IL3,	VL12,	VL23,	VL31,	INUT]])
    lr_pred = lr_pred[0]

    svc_pred = svc_classifier.predict([[OTI,	WTI,	ATI,	OLI,	OTI_A,	OTI_T, VL1,	VL2,	VL3,	IL1,	IL2,	IL3,	VL12,	VL23,	VL31,	INUT]])
    svc_pred = svc_pred[0]

    knn_pred = knn_classifier.predict([[OTI,	WTI,	ATI,	OLI,	OTI_A,	OTI_T, VL1,	VL2,	VL3,	IL1,	IL2,	IL3,	VL12,	VL23,	VL31,	INUT]])
    knn_pred = knn_pred[0]

    decision_pred = decision.predict([[OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT]])
    decision_pred = decision_pred[0]

# Random Forest
    random_forest_pred = random_forest.predict([[OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT]])
    random_forest_pred = random_forest_pred[0]

# Extra Trees
    et_pred = et.predict([[OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT]])
    et_pred = et_pred[0]

# AdaBoost
    ada_pred = ada.predict([[OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT]])
    ada_pred = ada_pred[0]

# XGBoost
    xgb_pred = xgb.predict([[OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT]])
    xgb_pred = xgb_pred[0]

    probability = ((lr_pred + svc_pred + knn_pred + decision_pred + random_forest_pred + et_pred + ada_pred + xgb_pred ) / 8.0) * 100.0
    probability = round(probability, 2)

    answer = ""

    if probability > 67.0:
        answer = "Very High Risk!"
    elif probability > 34.0:
        answer = "Moderate Risk!"
    elif probability > 1.0:
        answer = "Low Risk!"
    else:
        answer = "No Risk"

    return "Transformer Fault Probabilty: "+str(probability)+"% "+answer

def main():
    st.title("Transformer Fault Prediction using Machine Learning")

    st.subheader("Input Parameters for the Transformer")
    html_temp = """
    <div style="background-color:#ff6347; padding:10px; border-radius:10px;">
    <h3 style="color:white; text-align:center;">Transformer Parameters</h3>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        OTI = st.text_input("OTI", "Type Here")
        WTI = st.text_input("WTI", "Type Here")
        ATI = st.text_input("ATI", "Type Here")
        OLI = st.text_input("OLI", "Type Here")
        OTI_A = st.text_input("OTI_A", "Type Here")
        OTI_T = st.text_input("OTI_T", "Type Here")
    with col2:
        VL1 = st.text_input("VL1", "Type Here")
        VL2 = st.text_input("VL2", "Type Here")
        VL3 = st.text_input("VL3", "Type Here")
        IL1 = st.text_input("IL1", "Type Here")
        IL2 = st.text_input("IL2", "Type Here")
        IL3 = st.text_input("IL3", "Type Here")
    with col3:
        VL12 = st.text_input("VL12", "Type Here")
        VL23 = st.text_input("VL23", "Type Here")
        VL31 = st.text_input("VL31", "Type Here")
        INUT = st.text_input("INUT", "Type Here")

    st.write('<style>div.stButton > button {background-color: #4CAF50; color: white; font-size: 16px;}</style>', unsafe_allow_html=True)

    if st.button("Predict"):
        result = predict_probability(OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT)

    st.success(f'{result}')



if __name__=='__main__':
    main()

