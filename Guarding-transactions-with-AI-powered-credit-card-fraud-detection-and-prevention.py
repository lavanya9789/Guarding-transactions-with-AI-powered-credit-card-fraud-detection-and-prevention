import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
import streamlit as st
import uuid

# Set random seed for reproducibility
np.random.seed(42)

# Function to load and preprocess data
def load_and_preprocess_data(file_path='creditcard.csv'):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Check for missing values
    df.fillna(method='ffill', inplace=True)
    
    # Feature scaling
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    
    # Drop original columns
    df = df.drop(['Amount', 'Time'], axis=1)
    
    return df

# Function for exploratory data analysis
def perform_eda(df):
    st.subheader("Exploratory Data Analysis")
    
    # Class distribution
    st.write("Class Distribution (0: Non-Fraud, 1: Fraud)")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    st.pyplot(fig)
    
    # Correlation matrix
    st.write("Correlation Matrix")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Transaction amount distribution
    st.write("Scaled Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Amount_scaled'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

# Function to train and evaluate model
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Initialize model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train model
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluation
    st.subheader("Model Evaluation")
    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    st.write("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # ROC Curve
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    st.write("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    return rf_model

# Function for real-time prediction interface
def prediction_interface(model, scaler):
    st.subheader("Real-Time Fraud Detection")
    
    # Input fields for transaction features
    features = {}
    for i in range(1, 29):
        features[f'V{i}'] = st.number_input(f'Feature V{i}', value=0.0, format="%.6f")
    features['Amount'] = st.number_input('Transaction Amount', min_value=0.0, value=0.0)
    features['Time'] = st.number_input('Time (seconds)', min_value=0.0, value=0.0)
    
    if st.button("Predict"):
        # Prepare input data
        input_data = [features[f'V{i}'] for i in range(1, 29)]
        input_data.append(scaler.transform([[features['Amount']]])[0][0])
        input_data.append(scaler.transform([[features['Time']]])[0][0])
        
        # Make prediction
        prediction = model.predict([input_data])[0]
        probability = model.predict_proba([input_data])[0][1]
        
        # Display result
        if prediction == 1:
            st.error(f"Fraudulent Transaction Detected! (Probability: {probability:.2%})")
        else:
            st.success(f"Legitimate Transaction (Probability of Fraud: {probability:.2%})")

# Main function
def main():
    st.title("AI-Powered Credit Card Fraud Detection")
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_data()
    except FileNotFoundError:
        st.error("Dataset file 'creditcard.csv' not found. Please upload the Kaggle Credit Card Fraud Detection Dataset.")
        return
    
    # Perform EDA
    perform_eda(df)
    
    # Prepare features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Handle imbalanced data with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Train-test split
    X_train, X_test, axis=1)
    y = df['Class']
    
    # Handle imbalanced data with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Train and evaluate model
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Initialize scaler for prediction interface
    scaler = StandardScaler()
    scaler.fit(df[['Amount_scaled', 'Time_scaled']])
    
    # Prediction interface
    prediction_interface(model, scaler)

if __name__ == "__main__":
    main()
