# Enrollment Conversion Rate Optimization

This project involves building a machine learning model to predict whether a lead will convert based on various features. The model is deployed using a Streamlit web application that allows users to input Enrollment Conversion Rate Optimization data and get predictions in real time.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Streamlit App](#streamlit-app)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [License](#license)

## Project Overview

This project aims to predict whether a lead will convert based on its attributes, such as source, interest level, program offered, and the number of follow-ups. A Logistic Regression model is used for classification, and the model is served via a Streamlit application.

### Key Features
- Predict lead conversion based on various input features.
- One-hot encoding of categorical features.
- Display processed input data and prediction results.
- Visualization of the output in the form of an image.

## Dataset

The dataset used for training consists of the following columns:
- `Lead ID`: A unique identifier for each lead.
- `Source`: The source of the lead (e.g., website, email, social media).
- `Interest Level`: The level of interest (e.g., high, medium, low).
- `Contacted`: Whether the lead was contacted or not.
- `Follow Ups`: The number of follow-ups made to the lead.
- `Program Offered`: The program offered to the lead (e.g., Data Science, Web Development).

The target variable is `Conversion`, where 1 represents a converted lead and 0 represents a non-converted lead.

## Model Training

The model is trained using `Logistic Regression` and saved for later use in the Streamlit app.

### Steps Involved:
1. **Data Preprocessing**: The data is preprocessed by handling categorical columns using one-hot encoding.
2. **Model Training**: A Logistic Regression model is trained using the preprocessed data.
3. **Model Evaluation**: The model's accuracy is evaluated using test data.
4. **Model Saving**: The trained model is saved using the `joblib` library for future use in the application.

## Streamlit App

The Streamlit app allows users to input lead information and get predictions on whether the lead will convert. The app processes the input data, makes predictions using the trained model, and displays the results.

### Features of the Streamlit App:
- **Input Fields**: Users can enter lead details like source, interest level, contacted status, number of follow-ups, and program offered.
- **Prediction**: The app predicts the likelihood of conversion for the entered lead.
- **Processed Data Display**: The app displays the processed input data to ensure transparency.
- **Visualization**: The app shows a prediction result image.

## Requirements

To run this project, you need the following Python libraries:
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
## Output

![Output Image 1](https://github.com/minalmmm/.-Enrollment-Conversion-Rate-Optimization/blob/main/images/img1.png)
![Output Image 2](https://github.com/minalmmm/.-Enrollment-Conversion-Rate-Optimization/blob/main/images/img2.png)


