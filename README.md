# Sentiment Analysis of English Texts

This project is created by Team AIvengers as part of course project for DA241M Data Science and Artificial Intelligence Minor Jul-Nov 2024 IIT Guwahati.Team members who have contributed to this project are:
  1. Satyaki Ray(230123080), Sophomore, MnC
  2. Amanaganti Chethan Reddy(230102117), Sophomore, ECE
  3. Ponnekanti Bipan Chandra(230102072), Sophomore, ECE

## Project Overview

This project aims to analyze English texts/reviews/user-provided feedback, apply ML models on them and classify them as positive or negative statements based on the sentiment they are trying to convey. For this, we first obtained a movie reviews dataset from Kaggle and trained two separate models on the dataset.

[Link to the Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


This repository contains two jupyter notebooks that we have used for performing sentiment analysis. The project uses two ML models:
  1. **Multinomial Naive Bayes**
  2. **Linear Support Vector Machine(SVM)**
and classifies English texts as positive or negative.

Also, we have created a sentiment analysis web-app using **Streamlit** which takes user input and recognizes their true sentiments based on the models that we have trained. Since SVM has a higher accuracy over Naive Bayes, we chose to support our app by the SVM model to predict results more accurately.


# About the app: Critique's Sentiment

A sentiment analysis web app that predicts the sentiment of user-provided text reviews. 
This project is built using **Streamlit** and leverages comprehensive text preprocessing, TF-IDF vectorization, and machine learning models to classify reviews as positive or negative.

## Overview
Critique's Sentiment provides an accessible way to perform sentiment analysis on user-input text. The app uses preprocessing to clean and standardize text data, followed by vectorization and classification using a trained sentiment analysis model. With a simple interface, users can see instant sentiment results.

## Features
- **Real-time Sentiment Prediction**: Users receive instant feedback on whether a review is positive or negative.
- **Text Preprocessing Pipeline**: Detailed text processing steps such as lowercasing, stop word removal, lemmatization, and sentiment-aware tokenization.
- **ML Model for Classification**: A trained model(SVM) classifies text sentiment with high accuracy. 
- **TF-IDF Vectorization**: Converts text data into a format (allocates text data in vector space as numerical data) that machine learning models can utilize effectively.

 Project Structure
```plaintext
Critique-s-sentiment/
├── streamlit_app.py     # Main Streamlit app file
├── requirements.txt     # List of dependencies
├── svm.pkl              # SVM model pickle file
├── vectorizer1.pkl      # TF-IDF vectorizer pickle file
└── README.md            # Project README file
```

## Installation

To run the project locally, follow these steps:

1. **UnZip the File Using winrar**:<br>
    Download the Critique-s-sentiment folder, unzip it and open the Terminal from the folder.

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   ```

   - For **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - For **Windows**:
     ```bash
     venv\Scripts\activate
     ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```
**Note: If there are issues in loading the app, there might be an issue with the 'vectorizer1.pkl' file. We recommend to download the file from [here](https://drive.google.com/file/d/1rF1Zqorg1EbV57Zzh0oPdB8AKIX0Wb0O/view?usp=drive_link) and replace the original file, then re-run the app.**

## Usage
Once the app is running, open the local server link (usually http://localhost:8501) in your browser.
Enter a text review into the input box, and the app will output whether the sentiment is positive or negative.

## Requirements
- **Python 3.7+**
- Required Python packages are listed in `requirements.txt`.

## Preprocessing Steps
- Lowercasing text
- Stopwords removal
- Lemmatization
- Sentiment-aware tokenization
- TF-IDF Vectorization

## Model
The trained machine learning model used for this app is preloaded as pickle file and loaded when the app starts. We picked SVM classifier as it gave us good accuracy and results. It uses TF-IDF vectorized data which is fitted with the training data for sentiment classification.  So now when we give any new test input it will be tokenized and the vector space is already predetermined so it will get vectorized with respect to the inital fitted data.
