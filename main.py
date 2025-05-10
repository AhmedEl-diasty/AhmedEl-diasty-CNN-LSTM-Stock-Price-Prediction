import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from Model import build_cnn_lstm_model

# 1. تحميل البيانات
df = yf.download('AAPL', start='2012-01-01', end='2022-01-01')
df = df[['Close']]

# 2. تطبيع البيانات
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 3. تجهيز بيانات التسلسل الزمني
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # [samples, time_steps, features]

# 4. تقسيم البيانات
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. بناء النموذج وتدريبه
model = build_cnn_lstm_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# 6. التقييم والتنبؤ
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 7. عرض النتائج
plt.figure(figsize=(12,6))
plt.plot(actual, color='blue', label='Actual Price')
plt.plot(predictions, color='red', label='Predicted Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
