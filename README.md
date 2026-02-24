# Stock Price Prediction Using Machine Learning

This project focuses on predicting future stock prices using historical stock market data and machine learning techniques. The objective of the project is to analyze past stock price trends and estimate future closing prices using a regression model.

The project uses real stock market data downloaded from Kaggle in CSV format. The dataset includes stock price information such as Date and Closing Price. The Date column is converted into a numerical format so that it can be used as input for the machine learning model. The data is sorted chronologically to preserve the time-series nature of stock prices.

A Linear Regression model is trained using the historical closing prices to learn the relationship between time and stock price movement. The dataset is split into training and testing sets to evaluate the model. After training, the model is used to predict stock prices for a user-defined number of future days.

The project also includes data visualization using Matplotlib. Historical stock prices are plotted to show trends, and predicted future prices are displayed alongside historical data to provide a clear comparison between actual and predicted values.

To run this project, Python must be installed along with the required libraries: pandas, numpy, matplotlib, and scikit-learn. The user must provide the path to the Kaggle CSV file when prompted (for example, `archive/TCS.csv`). The program can be executed using `python Stock_Price_Prediction.py`.

This project demonstrates basic concepts of data preprocessing, regression modeling, time-based prediction, and data visualization. It is suitable for academic submissions, machine learning practice, and beginner-level portfolio projects.

Author: Amrutha Reddy
