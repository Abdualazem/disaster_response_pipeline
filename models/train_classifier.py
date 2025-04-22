import sys
import pandas as pd
import re
import nltk
#nltk.download('punkt') # Takes a lot of time to download
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib

def load_data(database_filepath):
    """
    Load data from SQLite database.

    Args:
        database_filepath (str): Filepath for the SQLite database.

    Returns:
        X (pd.Series): Feature data (messages).
        Y (pd.DataFrame): Target data (categories).
        category_names (list): List of category names for classification.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    #df = pd.read_sql_table('Message', engine)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Normalize, tokenize, remove stopwords, and lemmatize input text.

    Args:
        text (str): Input message string.

    Returns:
        clean_tokens (list): List of cleaned tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  # Normalize text
    tokens = word_tokenize(text)  # Tokenize text
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stopwords.words("english")
    ]
    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline and perform grid search.

    Returns:
        cv (GridSearchCV): Grid search model object.
    """
    # Define pipeline with vectorizer, transformer, and classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Define grid search parameters (can be expanded)
    parameters = {
        'clf__estimator__n_estimators': [50],
        'clf__estimator__min_samples_split': [2],
        'vect__max_df': [1.0],
        'tfidf__use_idf': [True]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=1) # Setting n_jobs=-1 brings error due to paralled processing
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and print classification report for each category.

    Args:
        model: Trained model object.
        X_test (pd.Series): Test features.
        Y_test (pd.DataFrame): Test targets.
        category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f"Category: {col}")
        print(classification_report(Y_test[col], Y_pred[:, i]))
        print("-" * 60)

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.

    Args:
        model: Trained model object.
        model_filepath (str): Filepath to save the pickle file.
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()