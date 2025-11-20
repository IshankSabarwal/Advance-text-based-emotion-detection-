import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load the main dataset
df_main = pd.read_csv("emotion_dataset_raw.csv")

# Load feedback dataset if available
if os.path.exists("new_feedback.csv"):
    df_feedback = pd.read_csv("new_feedback.csv")
    
    df_feedback = df_feedback[['text', 'emotion']]
else:
    df_feedback = pd.DataFrame(columns=['text', 'emotion'])

# Merge datasets
df_combined = pd.concat([df_main, df_feedback], ignore_index=True)

# **Data Augmentation : If feedback data is too small, we are duplicating it multiple time to have an impact on prediction probabilities.
if len(df_feedback) > 0:
    df_feedback_augmented = pd.concat([df_feedback] * 100, ignore_index=True)  # Repeat feedback 100 times
    df_combined = pd.concat([df_main, df_feedback_augmented], ignore_index=True)


df_combined = df_combined[['text', 'emotion']]

#Print dataset information.
print(f"✅ Original dataset size: {df_main.shape}")
print(f"📌 Feedback dataset size: {df_feedback.shape}")
print(f"📌 Final dataset size after merging & cleaning: {df_combined.shape}")

# Splitdata
X_train, y_train = df_combined["text"], df_combined["emotion"]

#code to create a pipeline
pipeline = Pipeline([
    ("tfidfvectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

#Training the pipeline
pipeline.fit(X_train, y_train)

#Saving the pipeline
joblib.dump(pipeline, "model/text_emotion.pkl")
print("✅ Model retrained with user feedback!")
