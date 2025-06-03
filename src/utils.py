import re
from typing import List, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, confusion_matrix
from tqdm.auto import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
tqdm.pandas()


def clean_text(text: Union[str, float]) -> str:
    """
    Perform basic text cleaning: lowercase, remove punctuation,
    and extra whitespace.

    Args:
        text (str or float): Input text.

    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def preprocess_text(text: Union[str, float]) -> List[str]:
    """
    Clean and tokenize text, removing stopwords.

    Args:
        text (str or float): Raw input text.

    Returns:
        List[str]: List of cleaned tokens.
    """
    if pd.isna(text):
        return []

    # Convert to lowercase
    text = str(text).lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]

    return filtered_tokens


def evaluate_model(
    X_train, X_val, y_train, y_val, model
) -> Tuple[float, float]:
    """
    Evaluate a classification model using log-loss
    on train and validation sets.

    Args:
        X_train: Training feature set.
        X_val: Validation feature set.
        y_train: Training labels.
        y_val: Validation labels.
        model: Fitted classifier with predict_proba() method.

    Returns:
        Tuple[float, float]: Log-loss on train and validation sets.
    """
    preds_proba_train = model.predict_proba(X_train)
    preds_proba_val = model.predict_proba(X_val)

    log_loss_train = log_loss(y_train, preds_proba_train)
    log_loss_val = log_loss(y_val, preds_proba_val)

    return log_loss_train, log_loss_val


def plot_confusion_matrix(y_true, y_pred) -> None:
    """
    Plot a normalized confusion matrix using seaborn heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, cmap='YlOrBr', fmt='.2f')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
