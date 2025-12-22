import streamlit as st
import pandas as pd
import pickle

with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.set_page_config(page_title="Personality Predictor", page_icon="🧠", layout="wide")

st.markdown("""
<div style='background-color: #ffe6e6; padding: 15px; border-radius: 10px'>
    <h1 style='color: #800000; text-align: center;'>Personality Prediction App</h1>
    <p style='text-align: center;'>Predict whether a person is Extrovert or Introvert based on their habits</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    Time_spent_Alone = st.number_input('🛋 Time spent alone (hours)', min_value=0.0)
    Social_event_attendance = st.number_input('🎉 Social event attendance (number of events)', min_value=0.0)
    Going_outside_input = st.selectbox('🚶 Going outside frequency', ['Never', 'Rarely', 'Sometimes', 'Often', 'Very often'])
    Friends_circle_size = st.number_input('👥 Friends circle size', min_value=0.0)

with col2:
    Stage_fear = st.selectbox('🎤 Stage fear', ['No', 'Yes'])
    Drained_after_socializing = st.selectbox('😴 Drained after socializing', ['No', 'Yes'])
    Post_frequency = st.number_input('📱 Post frequency (posts per month)', min_value=0.0)

going_outside_map = {'Never':0, 'Rarely':1, 'Sometimes':2, 'Often':3, 'Very often':4}
Going_outside = going_outside_map[Going_outside_input]

input_df = pd.DataFrame({
    'Time_spent_Alone': [Time_spent_Alone],
    'Stage_fear': [Stage_fear],
    'Social_event_attendance': [Social_event_attendance],
    'Going_outside': [Going_outside],
    'Drained_after_socializing': [Drained_after_socializing],
    'Friends_circle_size': [Friends_circle_size],
    'Post_frequency': [Post_frequency]
})

input_df = pd.get_dummies(input_df, columns=['Stage_fear', 'Drained_after_socializing'], drop_first=False)

for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]

input_scaled = scaler.transform(input_df)

if st.button('Predict'):
    pred_encoded = model.predict(input_scaled)[0]
    personality_map = {0: 'Extrovert', 1: 'Introvert'}
    prediction = personality_map[pred_encoded]
    st.markdown(f"""
    <div style='background-color: #ccffcc; padding: 15px; border-radius: 10px'>
        <h2 style='color: #006600; text-align: center;'>Predicted Personality: {prediction}</h2>
    </div>
    """, unsafe_allow_html=True)

