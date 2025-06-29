import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Stress Level Prediction",
    page_icon="🧠",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧠 Stress Level Prediction System")
st.markdown("""
This application predicts stress levels based on various health and lifestyle factors.
The prediction is made on a scale of 1-10, where:
- 1-3: Low Stress
- 4-7: Moderate Stress
- 8-10: High Stress
""")

# Function to load and prepare data
@st.cache_data
def load_data():
    # Load your dataset here
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    
    # Handle missing values
    df = df.fillna("Nothing")
    df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")
    
    # Split blood pressure
    df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True)
    df[['Systolic BP', 'Diastolic BP']] = df[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric, errors='coerce')
    
    # Drop unnecessary columns
    df = df.drop(['Blood Pressure', 'Person ID', 'Sleep Disorder', 
                  'Physical Activity Level', 'Diastolic BP', 'Quality of Sleep'], axis=1)
    
    # Encode categorical variables
    le = LabelEncoder()
    cat_cols = ['Gender', 'Occupation', 'BMI Category']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    return df, le

# Load data and train model
try:
    df, label_encoder = load_data()
    X = df.drop('Stress Level', axis=1)
    y = df['Stress Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create input form
    st.header("📝 Enter Your Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_encoded = 1 if gender == "Male" else 0
        
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        
        occupation_options = {
            'Scientist': 0,
            'Doctor': 1,
            'Accountant': 2,
            'Teacher': 3,
            'Manager': 4,
            'Engineer': 5,
            'Sales Representative': 6,
            'Salesperson': 7,
            'Lawyer': 8,
            'Software Engineer': 9,
            'Nurse': 10
        }
        occupation = st.selectbox("Occupation", list(occupation_options.keys()))
        occupation_encoded = occupation_options[occupation]
        
        sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 24.0, 7.0, 0.1)
    
    with col2:
        bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight"])
        bmi_encoded = {"Underweight": 1, "Normal": 2, "Overweight": 3}[bmi_category]
        
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=60, max_value=120, value=75)
        daily_steps = st.number_input("Daily Steps", min_value=2000, max_value=25000, value=8000)
        systolic_bp = st.number_input("Systolic Blood Pressure", min_value=90, max_value=180, value=120)
    
    # Make prediction
    if st.button("Predict Stress Level"):
        input_data = np.array([[gender_encoded, age, occupation_encoded, sleep_duration,
                                bmi_encoded, heart_rate, daily_steps, systolic_bp]])
        
        prediction = model.predict(input_data)[0]
        
        # Define stress level category
        if prediction <= 3:
            category = "Low"
            color = "green"
        elif prediction <= 7:
            category = "Moderate"
            color = "orange"
        else:
            category = "High"
            color = "red"
        
        # Display prediction
        st.markdown(f"""
            <div class="prediction-box" style="background-color: {color}20;">
                <h2 style="color: {color};">Predicted Stress Level: {prediction}</h2>
                <h3>Category: {category} Stress</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Display recommendations based on stress level
        st.subheader("📋 Recommendations")
        if category == "Low":
            st.success("""
                - Maintain your current healthy lifestyle
                - Continue with regular exercise and good sleep habits
                - Practice preventive stress management techniques
            """)
        elif category == "Moderate":
            st.warning("""
                - Consider increasing your sleep duration
                - Add relaxation techniques to your daily routine
                - Review your work-life balance
                - Consider regular exercise if not already doing so
            """)
        else:
            st.error("""
                - Strongly consider consulting a healthcare professional
                - Implement stress reduction techniques immediately
                - Review and adjust your work schedule if possible
                - Prioritize sleep and physical activity
                - Consider meditation or mindfulness practices
            """)
        
        # Feature importance plot
        st.subheader("📊 Factors Influencing Your Stress Level")
        feature_importance = pd.DataFrame({
            'Feature': ['Gender', 'Age', 'Occupation', 'Sleep Duration', 
                       'BMI Category', 'Heart Rate', 'Daily Steps', 'Systolic BP'],
            'Importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=True)
        
        st.bar_chart(feature_importance.set_index('Feature'))

except Exception as e:
    st.error(f"""
        ⚠️ An error occurred: {str(e)}
        
        Please ensure that:
        1. The dataset file 'Sleep_health_and_lifestyle_dataset.csv' is in the same directory
        2. All required libraries are installed
        3. The data format matches the expected structure
    """)
