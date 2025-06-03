# Stroke Prediction Using Machine Learning

This project aims to build a machine learning model that can predict the likelihood of a patient having a stroke based on several health and lifestyle features or attributes. The provided Jupyter Notebook (`Stroke.ipynb`) covers the entire process from data collection, exploratory data analysis (EDA), data preprocessing, model building, to model evaluation.

## Project Summary

The dataset used is the "Full-Filled Brain Stroke Dataset" from Kaggle. The machine learning models explored are Random Forest and Gradient Boosting. Since the dataset is imbalanced, an oversampling technique was applied to improve model performance on the minority class (stroke patients).

## Notebook Structure

The Jupyter Notebook (`Stroke.ipynb`) consists of the following steps:

1.  **Module Installation**:
    * Installing the `kaggle` library to download the dataset.
    * Installing the `imblearn` library to handle imbalanced data.

2.  **Library Imports**:
    * Importing standard libraries for data analysis and machine learning such as `tensorflow`, `pandas`, `seaborn`, `numpy`, `matplotlib`.
    * Importing specific modules from `sklearn` for data splitting, models (RandomForestClassifier, GradientBoostingClassifier), and evaluation metrics.
    * Importing `RandomOverSampler` from `imblearn`.

3.  **Environment Setup**:
    * Verifying TensorFlow version and ensuring GPU is enabled for faster computation in Google Colab.

4.  **Data Collection**:
    * Uploading the Kaggle API token (`kaggle.json`).
    * Downloading the "Full-Filled Brain Stroke Dataset" from Kaggle.
    * Extracting the dataset file.

5.  **Loading and Understanding Data**:
    * Reading the `full_data.csv` dataset using pandas.
    * Displaying dataset information (`.info()`) showing 4981 rows, 11 columns, with no missing values.
    * Displaying the first 5 rows (`.head()`) to get an overview of the data.
    * Description of features in the dataset:
        * `gender`: Gender of the patient ("Male", "Female", "Other").
        * `age`: Age of the patient.
        * `hypertension`: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension.
        * `heart_disease`: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease.
        * `ever_married`: Marital status ("No" or "Yes").
        * `work_type`: Type of work ("children", "Govt_job", "Never_worked", "Private" or "Self-employed").
        * `Residence_type`: Type of residence ("Rural" or "Urban").
        * `avg_glucose_level`: Average glucose level in blood.
        * `bmi`: Body mass index.
        * `smoking_status`: Smoking status ("formerly smoked", "never smoked", "smokes" or "Unknown").
        * `stroke`: Target variable (1 if the patient had a stroke or 0 if not).

6.  **Exploratory Data Analysis (EDA)**:
    * Displaying descriptive statistics for numerical columns (`.describe()`).
    * Visualizing the target variable (`stroke`) distribution using seaborn's `countplot`, which shows data imbalance (majority non-stroke).
    * Analyzing and visualizing data of stroke patients based on:
        * Gender (`gender`).
        * Work type (`work_type`).
        * Smoking status (`smoking_status`).
        * Residence type (`Residence_type`).
        * Marital status (`ever_married`).
        * Age distribution (`age`) using `histplot`.
    * Creating a correlation heatmap between numerical features.

7.  **Feature Engineering**:
    * Converting categorical columns `ever_married` and `gender` to numerical (0 or 1).
    * Applying one-hot encoding to other categorical columns (`work_type`, `Residence_type`, `smoking_status`) using `pd.get_dummies()`.

8.  **Feature and Target Separation**:
    * Separating features (X) and target (y, i.e., the `stroke` column).
    * Splitting the data into training and testing sets (33% for the test set) using `train_test_split`.

9.  **Model Building (Machine Learning)**:
    * **Initial Model (Without Oversampling)**:
        * **Random Forest Classifier**:
            * Trained with parameters `n_estimators=300`, `criterion='gini'`, `max_depth=5`, `n_jobs=-1`, `random_state=0`.
            * Predictions made on the test data.
            * Evaluation using confusion matrix, classification report, and F1-score. The F1-score was very low due to data imbalance.
        * **Gradient Boosting Classifier**:
            * Trained with parameters `n_estimators=300`, `max_depth=5`, `random_state=0`.
            * Predictions made on the test data.
            * Evaluation using confusion matrix, classification report, and F1-score. The F1-score was also low.
    * **Handling Imbalanced Data (Oversampling)**:
        * Using `RandomOverSampler` with `sampling_strategy='not majority'` to balance the dataset.
        * Re-splitting the oversampled data into training (X\_train\_rs, y\_train\_rs) and testing (X\_test\_rs, y\_test\_rs) sets.
    * **Model with Oversampled Data**:
        * **Random Forest Classifier**:
            * The same `rf` model was retrained with the oversampled training data.
            * Predictions made on the oversampled test data.
            * Evaluation showed a significant improvement in the F1-score (around 0.818).
        * **Gradient Boosting Classifier**:
            * The same `gb` model was retrained with the oversampled training data.
            * Predictions made on the oversampled test data.
            * Evaluation showed very good performance with an F1-score of around 0.970 and no false negative predictions on the test data used.
    * **Model Comparison**:
        * Creating a DataFrame to compare the F1-scores of both models after oversampling. Gradient Boosting showed better performance.
    * **Feature Importance**:
        * Displaying feature importance from the Gradient Boosting model, which indicates that `age`, `avg_glucose_level`, and `bmi` are the most influential features.

10. **Model Saving**:
    * The Gradient Boosting model (`gb`) trained with oversampled data was saved to the file `gradient_boosting_model.pkl` using `joblib.dump()`.

## Results

The Gradient Boosting model trained using oversampled data showed the best performance with an F1-score of approximately 0.970 on the test data. This model successfully predicted stroke cases well, especially in minimizing false negatives. Features such as age, average glucose level, and BMI were identified as important factors in stroke prediction.

## Further Use

The saved model (`gradient_boosting_model.pkl`) can be used for deployment, for example, in a web application using Streamlit or Flask, to predict stroke for new patient data.

## How to Run the Notebook

1.  Ensure all libraries listed in the import section are installed.
2.  Prepare your `kaggle.json` file to download the dataset.
3.  Run the cells in the notebook sequentially.
