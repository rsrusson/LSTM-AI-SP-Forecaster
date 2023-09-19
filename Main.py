import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt


# Define time ranges for data collection
start = datetime(2000, 1, 1)
end = datetime.today()
start_training = datetime(2000,1,3)
end_one_day = end - timedelta(days=1)
end_one_month = end - timedelta(days=31)
end_three_month = end - timedelta(days=92)
end_six_month = end - timedelta(days=183)
end_twelve_month = end - timedelta(days=365)

day_shift = -1
month_shift = -23
three_shift = -64
six_shift = -129
twelve_shift = -255

# Column names for data frames
ind_column_names = ['GDP', 'UNRATE', 'CPIAUCNS', 'FEDFUNDS', 'PPIACO',
       'RSAFS', 'Open', 'Close', 'Volume']
dep_column_names = ['Next Day\'s Close', 'Next Month\'s Close', 'Three Month\'s Close', 'Six Month\'s Close', 'Twelve Month\'s Close']

# Fetch economic and market data
gdp = web.DataReader("GDP", "fred", start, end)
gdp_df = pd.DataFrame(gdp)
unemployment = web.DataReader("UNRATE", "fred", start, end)
unemployment_df = pd.DataFrame(unemployment)
interest_rates = web.DataReader("FEDFUNDS", "fred", start, end)
interest_rates_df = pd.DataFrame(interest_rates)
cpi = web.DataReader("CPIAUCNS", "fred", start, end)
cpi_df = pd.DataFrame(cpi)
ppi = web.DataReader("PPIACO", "fred", start, end)
ppi_df = pd.DataFrame(ppi)
retail_sales = web.DataReader("RSAFS", "fred", start, end)
retail_sales_df = pd.DataFrame(retail_sales)

ticker = "SPY"
spy_data = yf.download(ticker, start=start, end=end)
spy_df = pd.DataFrame(spy_data)

# Merge all fetched data into a single DataFrame
all_df = pd.merge(gdp_df, unemployment_df, left_index=True, right_index=True, how='outer')
all_df = pd.merge(all_df, interest_rates_df, left_index=True, right_index=True, how='outer')
all_df = pd.merge(all_df, cpi_df, left_index=True, right_index=True, how='outer')
all_df = pd.merge(all_df, ppi_df, left_index=True, right_index=True, how='outer')
all_df = pd.merge(all_df, retail_sales_df, left_index=True, right_index=True, how='outer')
all_df = pd.merge(all_df, spy_df, left_index=True, right_index=True, how='outer')

# Drop unneeded columns and handle missing data
all_df.drop('High', axis=1, inplace=True)
all_df.drop('Low', axis=1, inplace=True)
all_df.drop('Adj Close', axis=1, inplace=True)
all_df.ffill(inplace=True)
all_df['Next Day\'s Close'] = all_df['Close'].shift(-1)
all_df['Next Month\'s Close'] = all_df['Close'].shift(-23)
all_df['Three Month\'s Close'] = all_df['Close'].shift(-64)
all_df['Six Month\'s Close'] = all_df['Close'].shift(-129)
all_df['Twelve Month\'s Close'] = all_df['Close'].shift(-255)
all_df.index.name = 'Date'
all_df = all_df[1:]

sc = MinMaxScaler()
scaled_all_df = pd.DataFrame(sc.fit_transform(all_df), columns=all_df.columns, index=all_df.index)

sc_for_close = MinMaxScaler()
sc_for_close.fit(all_df[['Close']])

'''day_train_df = scaled_all_df[:day_shift]
X_day_df = day_train_df[ind_column_names].values
y_day_df = day_train_df[dep_column_names[0]].values
X_day_df = np.reshape(X_day_df, (X_day_df.shape[0], X_day_df.shape[1], 1))
day_train_size = int(len(X_day_df) * 0.8)
X_train_day, X_test_day = X_day_df[:day_train_size], X_day_df[day_train_size:]
y_train_day, y_test_day = y_day_df[:day_train_size], y_day_df[day_train_size:]
day_model = Sequential()
day_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_day.shape[1], 1)))
day_model.add(LSTM(units=50))
day_model.add(Dense(1))
day_model.compile(optimizer='adam', loss='mean_squared_error')
day_model.fit(X_train_day, y_train_day, epochs=50, batch_size=32)
day_predicted_y = day_model.predict(X_test_day)
day_predicted_y_unscaled = sc_for_close.inverse_transform(day_predicted_y)
actual_day_close_prices = all_df['Close'][day_train_size:].values[:len(day_predicted_y_unscaled)]

rmse_day = math.sqrt(mean_squared_error(actual_day_close_prices, day_predicted_y_unscaled))
mape_day = mean_absolute_percentage_error(actual_day_close_prices, day_predicted_y_unscaled)

tomorrow_data = scaled_all_df[ind_column_names].iloc[-1].values
tomorrow_data = tomorrow_data.reshape(1, tomorrow_data.shape[0], 1)
tomorrow_prediction_scaled = day_model.predict(tomorrow_data)
tomorrow_prediction_unscaled = sc_for_close.inverse_transform(tomorrow_prediction_scaled)

print(rmse_day, mape_day, tomorrow_prediction_unscaled)

day_fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title('Day Model')
ax.set_xlabel('Date', fontsize=18)
ax.set_ylabel('Close Price USD ($)', fontsize=18)
ax.plot(actual_day_close_prices, label='Actual')
ax.plot(day_predicted_y_unscaled, label='Predicted')
ax.legend(loc='lower right')
day_fig.show()'''

month_train_df = scaled_all_df[:month_shift]
X_month_df = month_train_df[ind_column_names].values
y_month_df = month_train_df[dep_column_names[1]].values
X_month_df = np.reshape(X_month_df, (X_month_df.shape[0], X_month_df.shape[1], 1))
month_train_size = int(len(X_month_df) * 0.8)
X_train_month, X_test_month = X_month_df[:month_train_size], X_month_df[month_train_size:]
y_train_month, y_test_month = y_month_df[:month_train_size], y_month_df[month_train_size:]
month_model = Sequential()
month_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_month.shape[1], 1)))
month_model.add(LSTM(units=50))
month_model.add(Dense(1))
month_model.compile(optimizer='adam', loss='mean_squared_error')
month_model.fit(X_train_month, y_train_month, epochs=50, batch_size=32)
month_predicted_y = month_model.predict(X_test_month)
month_predicted_y_unscaled = sc_for_close.inverse_transform(month_predicted_y)
month_future_df = scaled_all_df[ind_column_names].iloc[-23:]
month_future_df = np.reshape(month_future_df, (month_future_df.shape[0], month_future_df.shape[1], 1))
month_predicted_future = month_model.predict(month_future_df)
month_predicted_future_unscaled = sc_for_close.inverse_transform(month_predicted_future)
month_predicted_both_unscaled = np.concatenate((month_predicted_y_unscaled, month_predicted_future_unscaled), axis=0)

actual_month_close_prices = all_df['Close'][month_train_size:].values[:len(month_predicted_y_unscaled)]

last_60_actual = actual_month_close_prices[-37:]
last_60_predicted = month_predicted_both_unscaled[-60:]

rmse_month = math.sqrt(mean_squared_error(actual_month_close_prices, month_predicted_y_unscaled))
mape_month = mean_absolute_percentage_error(actual_month_close_prices, month_predicted_y_unscaled)


print(rmse_month, mape_month)

month_fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title('Month Model')
ax.set_xlabel('Date', fontsize=18)
ax.set_ylabel('Close Price USD ($)', fontsize=18)
ax.plot(last_60_actual, label='Actual')
ax.plot(last_60_predicted, label='Predicted')
ax.legend(loc='lower right')
month_fig.show()
plt.show()

'''three_train_df = scaled_all_df[:three_shift]
X_three_df = three_train_df[ind_column_names].values
y_three_df = three_train_df[dep_column_names[2]].values

six_train_df = scaled_all_df[:six_shift]
X_six_df = six_train_df[ind_column_names].values
y_six_df = six_train_df[dep_column_names[3]].values'''

'''twelve_train_df = scaled_all_df[:twelve_shift]
X_twelve_df = twelve_train_df[ind_column_names].values
y_twelve_df = twelve_train_df[dep_column_names[4]].values
X_twelve_df = np.reshape(X_twelve_df, (X_twelve_df.shape[0], X_twelve_df.shape[1], 1))
twelve_train_size = int(len(X_twelve_df) * 0.8)
X_train_twelve, X_test_twelve = X_twelve_df[:twelve_train_size], X_twelve_df[twelve_train_size:]
y_train_twelve, y_test_twelve = y_twelve_df[:twelve_train_size], y_twelve_df[twelve_train_size:]
twelve_model = Sequential()
twelve_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_twelve.shape[1], 1)))
twelve_model.add(LSTM(units=50))
twelve_model.add(Dense(1))
twelve_model.compile(optimizer='adam', loss='mean_squared_error')
twelve_model.fit(X_train_twelve, y_train_twelve, epochs=50, batch_size=32)
twelve_predicted_y = twelve_model.predict(X_test_twelve)
twelve_predicted_y_unscaled = sc_for_close.inverse_transform(twelve_predicted_y)

actual_twelve_close_prices = all_df['Close'][twelve_train_size:].values[:len(twelve_predicted_y_unscaled)]

rmse_twelve = math.sqrt(mean_squared_error(actual_twelve_close_prices, twelve_predicted_y_unscaled))
mape_twelve = mean_absolute_percentage_error(actual_twelve_close_prices, twelve_predicted_y_unscaled)

print(rmse_twelve, mape_twelve)

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(actual_twelve_close_prices)
plt.plot(twelve_predicted_y_unscaled)
plt.legend(['Actual', 'Predicted'], loc='lower right')
plt.show()
'''
print(f'Test test')