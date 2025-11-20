import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
from googletrans import Translator

translator = Translator()

#loading the full pipeline (vectorizer + classifier)
pipe_lr = joblib.load("model/text_emotion.pkl")

# Emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Function to translate text to English
def translate_to_english(text):
    try:
        translated = translator.translate(text, src='auto', dest='en')
        return translated.text
    except Exception as e:
        print(f"⚠ Translation failed: {e}")
        return text

# Function to predict emotion
def predict_emotions(docx):
    translated_text = translate_to_english(docx)
    return pipe_lr.predict([translated_text])[0]

def get_prediction_proba(docx):
    translated_text = translate_to_english(docx)
    return pipe_lr.predict_proba([translated_text])

# Function to save user feedback.
def save_feedback(original_text, correct_emotion):
    feedback_file = "new_feedback.csv"
    
    # Translate to English before saving
    translated_text = translate_to_english(original_text)
    
    feedback_data = pd.DataFrame([[translated_text, correct_emotion]], columns=["text", "emotion"])
    
    #code to append new feedback to CSV
    if os.path.exists(feedback_file):
        feedback_data.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        feedback_data.to_csv(feedback_file, index=False)

#code to Initialize session state variables
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None

if "corrected_emotion" not in st.session_state:
    st.session_state.corrected_emotion = "neutral"  # Default value

#UI - done with streamlit library
def main():
    st.title("🌍 Multilingual Text Emotion Detection")
    st.subheader("Detect Emotions in Any Language!")

    #Input using session state.
    st.session_state.user_input = st.text_area("📝 Type Here (Any Language)", value=st.session_state.user_input)

    #Submit button implementation.
    if st.button("🔍 Detect Emotion"):
        if st.session_state.user_input.strip():
            translated_text = translate_to_english(st.session_state.user_input)
            st.session_state.prediction = predict_emotions(translated_text)
            st.session_state.probability = get_prediction_proba(translated_text)

    # Display
    if st.session_state.prediction:
        col1, col2 = st.columns(2)

        with col1:
            st.success("📜 Original Text")
            st.write(st.session_state.user_input)

            translated_text = translate_to_english(st.session_state.user_input)
            if translated_text != st.session_state.user_input:
                st.info("🌎 Translated to English")
                st.write(translated_text)

            st.success("🎭 Predicted Emotion")
            emoji_icon = emotions_emoji_dict.get(st.session_state.prediction, "❓")
            st.write(f"{st.session_state.prediction} {emoji_icon}")
            st.write(f"Confidence: {np.max(st.session_state.probability):.2f}")

        with col2:
            st.success("📊 Prediction Probability")
            proba_df = pd.DataFrame(st.session_state.probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotion", "Probability"]
            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Emotion', y='Probability', color='Emotion')
            st.altair_chart(fig, use_container_width=True)

        # Feedback section
        st.subheader("Was the prediction correct?")
        if st.button("Yes ✅"):
            st.success("Thank you for your feedback! 👍")

        #dropdown implementation
        selected_emotion = st.selectbox(
            "Choose the correct emotion:",
            options=list(emotions_emoji_dict.keys()),
            index=list(emotions_emoji_dict.keys()).index(st.session_state.corrected_emotion),
            key="emotion_selector"
        )

        #Session update code.
        if selected_emotion != st.session_state.corrected_emotion:
            st.session_state.corrected_emotion = selected_emotion

        #Submit without resetting the page.
        if st.button("Submit Correction ❌"):
            save_feedback(st.session_state.user_input, st.session_state.corrected_emotion)
            st.success("✅ Your correction has been saved! This will help improve the model.")
            st.session_state.prediction = st.session_state.corrected_emotion  # Update displayed emotion

if __name__ == '__main__':
    main()
