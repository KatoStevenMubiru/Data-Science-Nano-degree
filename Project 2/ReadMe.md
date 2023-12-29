
# Disaster Response Pipeline

## Project Summary

This project aims to analyze disaster data to build a model for an API that classifies disaster messages. The goal is to categorize messages so they can be sent to appropriate disaster relief agencies. The project includes an ETL pipeline to clean and process the data, an ML pipeline to train a multi-output classification model, and a Flask web app for user interaction.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
   - [Dependencies](#dependencies)
   - [Installing Dependencies](#installing-dependencies)
3. [Files in the Repository](#files-in-the-repository)
4. [Running the ETL Pipeline](#running-the-etl-pipeline)
5. [Running the ML Pipeline](#running-the-ml-pipeline)
6. [Running the Web App](#running-the-web-app)
7. [Additional Information](#additional-information)
   - [Example Commands](#example-commands)
   - [Web App Visualizations](#web-app-visualizations)
8. [Acknowledgements](#acknowledgements)

## Project Overview

In this project, we analyze disaster data from Appen (formerly Figure 8) to build a model for an API that classifies disaster messages. The project includes three main components:

1. **ETL Pipeline (process_data.py):** Loads, cleans, and stores data in an SQLite database.
2. **ML Pipeline (train_classifier.py):** Builds and trains a multi-output classification model using NLP techniques.
3. **Flask Web App (run.py):** Allows emergency workers to input a new message and get classification results.

## Getting Started

### Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-Learn
- NLTK
- Flask
- Plotly
- SQLAlchemy

### Installing Dependencies

```bash
pip install -r requirements.txt
```

## Files in the Repository

- **app:**
  - `run.py`: Flask web app.
  - **templates:**
    - `master.html`: Main page of the web app.
    - `go.html`: Classification result page.

- **data:**
  - `disaster_categories.csv`: Categories dataset.
  - `disaster_messages.csv`: Messages dataset.
  - `process_data.py`: ETL pipeline script.
  - `InsertDatabaseName.db`: SQLite database (output of the ETL pipeline).

- **models:**
  - `train_classifier.py`: ML pipeline script.
  - `classifier.pkl`: Saved model (output of the ML pipeline).

- `README.md`: Project documentation.

## Running the ETL Pipeline

```bash
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

## Running the ML Pipeline

```bash
python train_classifier.py DisasterResponse.db classifier.pkl
```

## Running the Web App

```bash
python run.py
```

Visit [http://localhost:3001/](http://localhost:3001/) in your web browser.

## Additional Information

### Example Commands

- For running the ETL pipeline:
  ```bash
  python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
  ```

- For running the ML pipeline:
  ```bash
  python train_classifier.py DisasterResponse.db classifier.pkl
  ```

### Web App Visualizations

- The web app displays visualizations based on data extracted from the SQLite database.

## Acknowledgements

This project is part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
```

