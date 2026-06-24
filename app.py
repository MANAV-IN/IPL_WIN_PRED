import os
import pickle
import pandas as pd
import streamlit as st

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickle_path = os.path.join(BASE_DIR, 'pipe.pkl')
pipe = pickle.load(open(pickle_path, 'rb'))

st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=0, step=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

if st.button('Predict Probability'):
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams cannot be the same!")
    else:
        runs_remaining = target - score
        balls_remains = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        if overs > 0:
            CRR = (score * 6) / (120 - balls_remains)
        else:
            CRR = 0.0

        if balls_remains > 0:
            RRR = (runs_remaining * 6) / balls_remains
        else:
            RRR = 0.0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_remaining': [runs_remaining],
            'balls_remains': [balls_remains],
            'wickets_left': [wickets_left],
            'total_runs_x': [target],
            'CRR': [CRR],
            'RRR': [RRR]
        })

        # Run inference
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display results nicely
        st.markdown("---")
        st.subheader("Win Probability")
        st.header(f"🏏 {batting_team}: {round(win * 100)}%")
        st.header(f"🍒 {bowling_team}: {round(loss * 100)}%")