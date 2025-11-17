import streamlit as st
import pandas as pd 
import numpy as np
#import time
import mlflow
import joblib


# Page configuration
st.set_page_config(
    page_title="Job Recommendation System"
)

# mlflow config
MLFLOW_TRACKING_URI = "file:./mlruns" # Changed to relative path for Docker
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# The user must update these URIs, e.g., 'runs:/<RUN_ID>/model_artifact_path'
MODEL_URI = '' 
VECTORIZER = ''
LABEL_ENCODER = ''

# Load the model
@st.cache_resource() # Uncommented to cache models for performance
def load_artifacts(model_uri, vectorizer_uri, encoder_uri):
    try:
        # Load artifacts via MLflow
        model = mlflow.sklearn.load_model(model_uri)
        vectorizer = mlflow.sklearn.load_model(vectorizer_uri)
        label_encoder = mlflow.sklearn.load_model(encoder_uri)
        st.success('Model loaded with MLflow')
        return model, vectorizer, label_encoder

    except Exception as e:
        # Fallback to joblib - paths changed to relative for Docker
        st.warning(f"MLflow loading failed: {e}. Falling back to local joblib files.")
        model = joblib.load('./models/xgb_job_title_recommender.pkl') 
        vectorizer = joblib.load('./models/skills_tfidf_vectorizer.pkl')  
        label_encoder = joblib.load('./models/job_title_label_encoder.pkl')  
        st.success('Model loaded with joblib')
        return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_artifacts(MODEL_URI, VECTORIZER, LABEL_ENCODER)
    
# prediction function
def predict_job_titles(skills, top_k=3):
    """_summary_

    Args:
        skills (string): User input of skills

    Returns:
        string: Predicted job titles with probabilities
    """
    
    # vectorize the input skills
    user_input = vectorizer.transform([skills]) 
    
    # make prediction
    probabilities = model.predict_proba(user_input)[0]
    
    # convert probabilities to job titles
    job_titles = label_encoder.inverse_transform(np.argsort(probabilities)[::-1][:top_k])
    top_probabilities = probabilities[np.argsort(probabilities)[::-1][:top_k]]
    
    return list(zip(job_titles, top_probabilities))

# app ui
st.title('Job Recommendation System')
st.write('Enter your skills, and our AI will suggest your top three job titles!')

#input
skills = st.text_area('Enter your skills (separated by commas):', height=75, placeholder='e.g., Python, Data Analysis, Machine Learning')

# button for predictions
if st.button('Get Job Recommendations'):
    if not skills.strip():
        st.warning('Please enter skills to get recommendations')
    else:
        skills = skills.lower()
        result = predict_job_titles(skills)
        
        # display results
        st.success('Done!')
        #st.metric(label='Top 3 Job Recommendations', value=result[0][0], delta=f'Probability: {result[0][1]:.2f}')
        # columns = st.columns(len(result))
        for i, (job_title, prob) in enumerate(result):
        #    with columns[i]:
        #        st.subheader(f"{i+1}. {job_title}")
        #        st.write(f"Probability: {prob:.2f}")
            st.metric(
                label=f"{i+1}. {job_title}",
                value=f"Probability: {prob:.2f}",
                delta='Confidence Score'
            )


#footer
st.markdown('---')