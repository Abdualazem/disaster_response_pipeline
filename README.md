# Disaster Response Pipeline Project

## Introduction

This project provides an end-to-end solution for processing, classifying, and visualising disaster response messages. It features:
- An ETL pipeline for data cleaning and storage
- A machine learning pipeline for multi-label text classification
- An interactive web application for message analysis

**Technologies used:**  
Python, scikit-learn, NLTK, Flask, Plotly, SQLAlchemy

**Project goal:**  
Help emergency teams quickly categorise and prioritise disaster-related messages for more effective response.

## Data Description

The dataset used in this project consists of real messages sent during disaster events provided by [Appen](https://www.appen.com/) (formerly known as Figure 8). Each message is labelled with one or more categories relevant to disaster response, such as requests for aid, infrastructure damage, or weather-related information. The data includes the text of the messages and metadata, such as the genre (e.g., direct, news, social).

This dataset enables the development of a machine learning pipeline that classifies incoming messages into multiple categories, helping emergency response teams prioritise and route messages efficiently. The accompanying web application allows users to input new messages for classification and offers interactive visualisations to explore the distribution and nature of the disaster response data.

## Getting Started

Follow these steps to prepare your data, train your model, and launch the web application.

### 1. Data Preparation

- **Clean and store data in the database:**

  ```sh
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  ```

### 2. Model Training

- **Train the classifier and save the model:**

  ```sh
  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  ```

### 3. Launch the Web Application

- **Start the web app from the app directory:**

  ```sh
  python run.py
  ```

- **Access the app:**  
  Open your browser and go to [http://127.0.0.1:3001](http://127.0.0.1:3001)

---

## Notes on Model Performance and Improvements

- The dataset contains imbalanced classes for several categories, which can lead to high accuracy but poor performance on less frequent classes.
- To address this, consider:
  - Setting `class_weight='balanced'` in your classifier.
  - Trying alternative algorithms such as logistic regression or gradient boosting.
  - Using resampling techniques (e.g., SMOTE or random undersampling) to balance the data.
  - Expanding your grid search to tune more hyperparameters.
  - Evaluating your model using metrics like precision, recall, and F1-score for each class.

---

**Tip:**  
Improving model fairness and performance for all categories may require experimenting with the above strategies and carefully analysing your results.

**Note:**  
The trained model file (`models/classifier.pkl`) is not included in this repository due to GitHub's file size limits.  

To use the application, you must train the model yourself using the instructions above, or download it from a shared link if provided.

## Acknowledgements

This project uses disaster response message data provided by [Appen](https://www.appen.com/) (formerly Figure 8). We would like to thank Appen for making this dataset publicly available, which enabled the development and demonstration of this project.  
Special thanks to the [Udacity Data Scientist Nanodegree programme](https://www.udacity.com/enrollment/nd025) for project guidance and resources.