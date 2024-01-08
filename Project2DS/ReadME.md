# Disaster Response Pipeline Project

## Project Summary
The Disaster Response Pipeline project is part of the Data Science Nanodegree Program by Udacity. The project involves analyzing disaster data from Figure Eight to build a model that classifies disaster messages. The project includes a web app where an emergency worker can input a new message and receive classification results in several categories. The web app also displays visualizations of the data.

## Project Components
There are three main components of the project:

1. **ETL Pipeline**: A data cleaning pipeline that:
   - Loads the `messages` and `categories` datasets
   - Merges the two datasets
   - Cleans the data
   - Stores it in a SQLite database

2. **ML Pipeline**: A machine learning pipeline that:
   - Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file

3. **Flask Web App**: A web application that:
   - Allows users to input disaster messages and receive classification results
   - Displays visualizations of the training dataset

## How to Run the Python Scripts
1. Run the following commands in the project's root directory to set up the database and model.

    - To run the ETL pipeline that cleans and stores the data in the database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains the classifier and saves the model:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the `app` directory to run the web app:
   `python run.py`

## How to Use the Web App
1. Go to http://0.0.0.0:3001/ (or the appropriate IP address and port number if different).
2. Input a disaster message into the text box and click "Classify Message".
3. View the classification results for the message across various categories.
4. Explore visualizations of the training data.

## File Descriptions
- `app/`
  - `templates/`: HTML templates for the web app.
    - `go.html`: Classification result page.
    - `master.html`: Main page with message input and visualizations.
  - `run.py`: Flask file to run the web app.

- `data/`
  - `disaster_categories.csv`: Dataset containing disaster categories.
  - `disaster_messages.csv`: Dataset containing disaster messages.
  - `DisasterResponse.db`: SQLite database containing cleaned data.
  - `process_data.py`: ETL pipeline script for processing and cleaning data.

- `models/`
  - `train_classifier.py`: Machine learning pipeline script for training and saving the classifier.

- `ETL Pipeline Preparation.ipynb`: Jupyter notebook for ETL pipeline preparation.
- `ML Pipeline Preparation.ipynb`: Jupyter notebook for ML pipeline preparation.

## Licensing, Authors, Acknowledgements
This project is part of the Data Science Nanodegree Program by Udacity. The data was provided by Figure Eight. Special thanks to Udacity for the project design and mentorship.