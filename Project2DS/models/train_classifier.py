import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    """
    Load data from the SQLite database.

    Parameters:
    database_filepath (str): The file path of the SQLite database.

    Returns:
    X (pandas.Series): The input features (messages).
    Y (pandas.DataFrame): The output labels (categories).
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('MessageCategories', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y

def tokenize(text):
    """
    Tokenize and lemmatize text data.

    Parameters:
    text (str): The text data to tokenize.

    Returns:
    clean_tokens (list of str): The list of clean tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline with GridSearchCV.

    Returns:
    cv (GridSearchCV): The GridSearchCV object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the model and print out the classification report for each category.

    Parameters:
    model (GridSearchCV or Pipeline): The trained model.
    X_test (pandas.Series): The test features.
    Y_test (pandas.DataFrame): The test labels.

    Returns:
    None
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(f'Category: {col}\n', classification_report(Y_test[col], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the model to a pickle file.

    Parameters:
    model (GridSearchCV or Pipeline): The trained model to save.
    model_filepath (str): The file path to save the model pickle.

    Returns:
    None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function to run the ML pipeline: load data, split dataset, build model,
    train model, evaluate model, and save model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()