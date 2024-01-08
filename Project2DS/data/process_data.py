#!/usr/bin/env python

import sys
import pandas as pd
from sqlalchemy import create_engine
import argparse

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets from CSV files and merge them.

    Parameters:
    messages_filepath (str): The file path of the messages CSV file.
    categories_filepath (str): The file path of the categories CSV file.

    Returns:
    df (pandas.DataFrame): The merged dataset containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Clean the merged dataset by splitting categories into separate columns, converting values to binary,
    and correcting the 'related' category to be binary (0 or 1).

    Parameters:
    df (pandas.DataFrame): The merged dataset to clean.

    Returns:
    df (pandas.DataFrame): The cleaned dataset with individual category columns.
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    
    # Convert 'related' column to binary
    categories['related'] = categories['related'].map(lambda x: 1 if x == 2 else x)
    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save the clean dataset into an SQLite database.

    Parameters:
    df (pandas.DataFrame): The cleaned dataset to save.
    database_filename (str): The file path for the SQLite database.

    Returns:
    None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('MessageCategories', engine, index=False, if_exists='replace')

def main():
    """
    Run the ETL pipeline: load data from CSV, clean data, and save data to SQLite database.
    """
    parser = argparse.ArgumentParser(description='ETL pipeline for disaster response data')
    parser.add_argument('messages_filepath', type=str, help='Filepath for the messages dataset')
    parser.add_argument('categories_filepath', type=str, help='Filepath for the categories dataset')
    parser.add_argument('database_filepath', type=str, help='Filepath for the SQLite database to store the cleaned data')
    
    args = parser.parse_args()

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(args.messages_filepath, args.categories_filepath))
    df = load_data(args.messages_filepath, args.categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(args.database_filepath))
    save_data(df, args.database_filepath)

    print('Cleaned data saved to database!')

if __name__ == '__main__':
    main()