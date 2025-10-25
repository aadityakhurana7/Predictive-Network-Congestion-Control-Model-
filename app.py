import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Network Congestion Detector", layout="wide")
st.title("🚦 Network Congestion Detection App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    data['congestion'] = data['IPv4 bytes'].apply(lambda x: 1 if x > 1e10 else 0)

    features = data.drop('congestion', axis=1)
    noise = np.random.normal(0, 1, features.shape)
    features_noisy = features + noise

    data_noisy = features_noisy.copy()
    data_noisy['congestion'] = data['congestion']

    X = data_noisy.drop('congestion', axis=1)
    y = data_noisy['congestion']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    axs[0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axs[0].plot([0, 1], [0, 1], linestyle='--')
    axs[0].set_title('ROC Curve')
    axs[0].legend()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[1])
    axs[1].set_title('Confusion Matrix')

    axs[2].plot(recall, precision)
    axs[2].set_title('Precision-Recall Curve')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("👈 Please upload a CSV file to start.")