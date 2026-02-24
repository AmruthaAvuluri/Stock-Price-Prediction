# Task 7: Stock Price Prediction using Real Kaggle Data
# Tools: pandas, matplotlib, sklearn (Linear Regression)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
file_name = input("Enter Kaggle stock CSV file name (e.g., TCS.csv): ")

df = pd.read_csv(file_name)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Sort data
df = df.sort_values('Date')

print("\nDataset Loaded Successfully!")
print(df.head())

# ---------------------------------------------------
# VISUALIZE HISTORICAL TREND
# ---------------------------------------------------
plt.figure()
plt.plot(df['Date'], df['Close'])
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("Historical Stock Prices")
plt.show()

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
# Convert Date to numeric
df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

X = df[['Date_ordinal']]
y = df['Close']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# ---------------------------------------------------
# FUTURE PREDICTION
# ---------------------------------------------------
future_days = int(input("Enter number of future days to predict: "))

last_date = df['Date'].max()

future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days + 1)]
future_dates_ordinal = [[date.toordinal()] for date in future_dates]

future_prices = model.predict(future_dates_ordinal)

# ---------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------
print("\nPredicted Future Prices:")
for date, price in zip(future_dates, future_prices):
    print(f"{date.date()} -> {price:.2f}")

# ---------------------------------------------------
# VISUALIZE PREDICTION
# ---------------------------------------------------
plt.figure()
plt.plot(df['Date'], df['Close'], label="Historical Prices")
plt.plot(future_dates, future_prices, linestyle='dashed', label="Predicted Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("Stock Price Prediction (Kaggle Data)")
plt.legend()
plt.show()


# as i take csv as archive form kaggle after in output page it geats like 
# Enter Kaggle stock CSV file name (e.g., TCS.csv): make sure to enter "archive/TCS.csv"
# as my csv in archive

