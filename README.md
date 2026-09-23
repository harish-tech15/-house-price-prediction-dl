# 🏠 House Price Prediction using Deep Learning

A Deep Learning based House Price Prediction application built using
TensorFlow, Keras and Streamlit.

## 🚀 Project Overview

This project uses an Artificial Neural Network (ANN) to predict
median house values based on housing and geographical features.

## 🧠 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Keras
- Streamlit

## 📊 Features

The model uses the following features:

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

## 🤖 Deep Learning Architecture

- Dense Layer: 128 neurons
- Dropout: 20%
- Dense Layer: 64 neurons
- Dropout: 20%
- Dense Layer: 32 neurons
- Dense Layer: 16 neurons
- Output Layer

## 📈 Model Evaluation

The model is evaluated using:

- MAE
- MSE
- RMSE
- R² Score

## 🌐 Streamlit Application

The trained model is integrated into a Streamlit web application where
users can enter housing information and receive an estimated house value.

## 📁 Project Structure

```text
house-price-prediction-dl/
│
├── app.py
├── requirements.txt
├── README.md
│
└── model/
    ├── house_price_model.keras
    ├── scaler.pkl
    └── feature_names.pkl
