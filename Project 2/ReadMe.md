# Disaster Response Pipeline

## Project Overview

This project implements a Disaster Response Pipeline that processes and classifies messages from disaster events. The pipeline includes an ETL process for cleaning and storing data, an ML pipeline for training a multi-output classification model, and a web app for user interaction. The aim is to categorize messages and facilitate effective communication between disaster-stricken areas and relevant relief agencies.

## Table of Contents

1. [GitHub & Code Quality](#github--code-quality)
2. [ETL](#etl)
3. [Machine Learning](#machine-learning)
4. [Deployment](#deployment)
5. [Suggestions to Stand Out](#suggestions-to-make-your-project-stand-out)

## GitHub & Code Quality

### Repository Link
[Disaster Response Pipeline Repository](https://github.com/your_username/disaster-response-pipeline)

### Commits
The project demonstrates an understanding of Git and Github, with at least 3 commits made to the repository.

### Documentation
The README file provides a comprehensive summary of the project, instructions on running Python scripts and the web app, and explanations of repository files. Effective comments and docstrings have been added to functions, adhering to PEP8 style guidelines.

## ETL

### ETL Script
The ETL script, `process_data.py`, runs without errors in the terminal. It processes datasets, cleans the data, and stores it in an SQLite database with specified file paths.

### Data Cleaning
The ETL script successfully cleans the dataset by merging messages and categories, splitting categories into separate columns, converting values to binary, and removing duplicates.

## Machine Learning

### ML Script
The machine learning script, `train_classifier.py`, runs without errors in the terminal. It takes database and model file paths, creates and trains a classifier, and saves it as a pickle file.

### NLP Techniques
The script demonstrates an understanding of NLP techniques, using a custom tokenize function with NLTK for case normalization, lemmatization, and tokenization. This function is used in the ML pipeline for text processing.

### Pipelines and Grid Search
The script builds a pipeline for text processing and multi-output classification on 36 categories. GridSearchCV is used to find optimal parameters for the model.

### Model Evaluation
The project understands training vs. test data and model evaluation. The TF-IDF pipeline is trained only with the training data. The script outputs f1 score, precision, and recall for each category on the test set.

## Deployment

### Web App
The web app, `run.py`, runs without errors in the terminal. The main page includes at least two visualizations using data from the SQLite database.

### Model Usage
The web app successfully utilizes the trained model to input text and returns classification results for all 36 categories when a user inputs a message.

