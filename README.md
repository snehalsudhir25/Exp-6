# Experiment 6: ARIMA Model for Forecasting

## Aim

To evaluate predictive models using performance metrics.

## Algorithm

1.  **START**
2.  Prepare stationary time series.
3.  Identify ARIMA -- (p, d, q) parameters using ACF/PACF.
4.  Fit the ARIMA model on the data.
5.  Forecast future periods and plot the results.
6.  Evaluate forecast accuracy.
7.  **STOP**



## Program

    import numpy as np
    import matplotlib.pyplot as plt
    
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    \# Generate stationary time series data
    np.random.seed(42)
    
    data = [0]
    
    for i in range(1, 100):
        value = 0.6 * data[i-1] + np.random.normal(0, 1)
        data.append(value)
    
    \# Convert into NumPy array
    data = np.array(data)
    
    \# Split training and testing data
    train = data[:80]
    test = data[80:]
    
    \# Plot ACF and PACF
    plot_acf(train, lags=20)
    plt.title("Autocorrelation Function (ACF)")
    plt.show()
    
    plot_pacf(train, lags=20)
    plt.title("Partial Autocorrelation Function (PACF)")
    plt.show()
    
    \# Fit ARIMA model
    model = ARIMA(train, order=(1, 0, 1))
    model_fit = model.fit()
    
    \# Forecast future values
    forecast = model_fit.forecast(steps=len(test))
    
    \# Evaluate model
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    rmse = np.sqrt(mse)
    
    print("ARIMA Model Evaluation")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    
    \# Plot actual and predicted values
    plt.figure(figsize=(10, 5))
    
    plt.plot(range(len(train)), train, label="Training Data")
    plt.plot(range(80, 100), test, label="Actual Values")
    plt.plot(range(80, 100), forecast, label="Forecast Values")
    
    plt.title("ARIMA Time Series Forecasting")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.show()

## Output

<img width="857" height="602" alt="image" src="https://github.com/user-attachments/assets/803ec206-601a-499d-8dde-8c0334d96189" />
<img width="708" height="546" alt="image" src="https://github.com/user-attachments/assets/20973343-382b-422b-8ed4-a6477017b7f8" />
<img width="853" height="547" alt="image" src="https://github.com/user-attachments/assets/c8054246-b289-41ca-9420-e535eb290948" />
<img width="1066" height="582" alt="image" src="https://github.com/user-attachments/assets/8b5a3559-0722-4795-8105-d11eedafde87" />




## Result
Thus, the ARIMA time-series forecasting model was successfully implemented using Python. A stationary time series was prepared, ARIMA parameters were selected, and future values were forecasted. Model performance was evaluated using MAE, MSE and RMSE.
