import pickle
import numpy as np
import streamlit as st
from sklearn.ensemble import VotingClassifier
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load the trained ensemble model from the pickle file
with open("ensemble_model.pkl", "rb") as f:
    ensemble_model = pickle.load(f)


def predict_probability(OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT):
    # Convert inputs to float
    # OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT = map(float, [OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT])
    input = np.array([[OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT]]).astype(np.float64)
    # Make predictions using ensemble voting
    prediction = ensemble_model.predict(input)[0]

    # probabilities = ensemble_model.predict_proba(input_values)[0]
    
    # Probability of being faulty
    # fault_probability = probabilities[1] * 100

    return f"Transformer Fault Prediction: {'Faulty' if prediction == 1 else 'Non-Faulty'}"
# Streamlit UI
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

    session_state = st.session_state
    if not hasattr(session_state, 'dummy_data'):
        session_state.dummy_data = {}

    if st.button("Generate Dummy Data"):
        session_state.dummy_data = generate_dummy_data()

    with col1:
        session_state.dummy_data["OTI"] = st.text_input("OTI", value=session_state.dummy_data.get("OTI", ""))
        session_state.dummy_data["WTI"] = st.text_input("WTI", value=session_state.dummy_data.get("WTI", ""))
        session_state.dummy_data["ATI"] = st.text_input("ATI", value=session_state.dummy_data.get("ATI", ""))
        session_state.dummy_data["OLI"] = st.text_input("OLI", value=session_state.dummy_data.get("OLI", ""))
        session_state.dummy_data["OTI_A"] = st.text_input("OTI_A", value=session_state.dummy_data.get("OTI_A", ""))
        session_state.dummy_data["OTI_T"] = st.text_input("OTI_T", value=session_state.dummy_data.get("OTI_T", ""))
    with col2:
        session_state.dummy_data["VL1"] = st.text_input("VL1", value=session_state.dummy_data.get("VL1", ""))
        session_state.dummy_data["VL2"] = st.text_input("VL2", value=session_state.dummy_data.get("VL2", ""))
        session_state.dummy_data["VL3"] = st.text_input("VL3", value=session_state.dummy_data.get("VL3", ""))
        session_state.dummy_data["IL1"] = st.text_input("IL1", value=session_state.dummy_data.get("IL1", ""))
        session_state.dummy_data["IL2"] = st.text_input("IL2", value=session_state.dummy_data.get("IL2", ""))
        session_state.dummy_data["IL3"] = st.text_input("IL3", value=session_state.dummy_data.get("IL3", ""))
    with col3:
        session_state.dummy_data["VL12"] = st.text_input("VL12", value=session_state.dummy_data.get("VL12", ""))
        session_state.dummy_data["VL23"] = st.text_input("VL23", value=session_state.dummy_data.get("VL23", ""))
        session_state.dummy_data["VL31"] = st.text_input("VL31", value=session_state.dummy_data.get("VL31", ""))
        session_state.dummy_data["INUT"] = st.text_input("INUT", value=session_state.dummy_data.get("INUT", ""))

    
    if st.button("Predict"):
        if not all(session_state.dummy_data.values()):
            st.warning("Please fill in all input fields.")
            return

        result = predict_probability(**session_state.dummy_data)
        st.success(result)

    

def generate_dummy_data():
    return {
        "OTI": np.random.uniform(-5.0, 250.0),
        "WTI": np.random.uniform(0.0, 1.0),
        "ATI": np.random.uniform(6.0, 80.0),
        "OLI": np.random.uniform(-20.0, 120.0),
        "OTI_A": 0.0,
        "OTI_T": 0.0,
       
        "VL1": np.random.uniform(5.6, 500.2),
        "VL2": np.random.uniform(5.4, 500.3),
        "VL3": np.random.uniform(5.4, 500.3),
        "IL1": np.random.uniform(0.0, 199.0),
        "IL2": np.random.uniform(0.0, 180.0),
        "IL3": np.random.uniform(0.0, 216.3),
        "VL12": np.random.uniform(0.0, 446.5),
        "VL23": np.random.uniform(0.0, 444.8),
        "VL31": np.random.uniform(0.0, 447.3),
        "INUT": np.random.uniform(0.0, 100.1),
    }

if __name__ == '__main__':
    main()
