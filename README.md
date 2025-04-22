# Disaster Response Pipeline Project

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
Improving model fairness and performance for all categories may require experimenting with the above strategies and carefully analyzing your results.