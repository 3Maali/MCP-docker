import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import ARIMA
from darts.metrics import mape
import warnings
import darts

# Check darts version for compatibility
DARTS_VERSION = darts.__version__
if DARTS_VERSION >= '0.30.0':
    TS_DF_METHOD = 'to_dataframe'
else:
    TS_DF_METHOD = 'pd_dataframe'

warnings.filterwarnings('ignore')

def load_data(data_path='./data/Final_Daily_Umrah_Statistics_2024__Tawaf__Saei__Other_.csv'):
    """
    Load and preprocess the Umrah statistics dataset.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with Date as index.
    """
    try:
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def generate_forecast(language='English'):
    """
    Generate a 7-day crowd forecast for Tawaf, Saei, and Other using the best model (ARIMA).

    Args:
        language (str): Language for crowd level labels ('Arabic' or 'English').

    Returns:
        pd.DataFrame: Forecast DataFrame with Date, Total_Predicted, Tawaf, Saei, Other, and crowd levels.
    """
    try:
        # Load data
        df = load_data()
        series = TimeSeries.from_dataframe(df, value_cols='Total', freq='D')

        # Split data
        train = series[:-7]
        val = series[-7:]

        # Initialize models
        models = [
            ARIMA(p=1, d=0, q=1, seasonal_order=(1, 0, 1, 7))
        ]

        forecasts = {}
        metrics = {}

        # Train and evaluate models
        for model in models:
            model_name = model.__class__.__name__
            if model_name == 'ARIMA':
                diff_series = series.diff()[1:]  # Differencing and remove NaN
                train_diff = diff_series[:-7]
                model.fit(train_diff)
                forecast_diff = model.predict(len(val))
                forecast_values = forecast_diff.values().flatten()
                if np.isnan(forecast_values).any():
                    print(f"{model_name}: Forecast contains NaN values")
                    continue
                last_value = train[-1].values().flatten()[0]
                forecast = last_value + forecast_values.cumsum()
                forecast_df = pd.DataFrame({
                    'Total': forecast,
                    'Date': val.time_index
                })
                forecast_df = forecast_df.set_index('Date')
                forecast = TimeSeries.from_dataframe(forecast_df, freq='D')
            else:
                model.fit(train)
                forecast = model.predict(len(val))

            forecast_vals = forecast.values().flatten()
            if np.isnan(forecast_vals).any():
                print(f"{model_name}: Forecast contains NaN values")
                continue

            error_mape = mape(val, forecast)
            forecasts[model_name] = forecast
            metrics[model_name] = {'mape': error_mape}

        if not metrics:
            raise ValueError("No valid forecasts generated due to NaN values")

        # Select best model
        best_model = min(metrics, key=lambda x: metrics[x]['mape'])
        best_model_obj = next(model for model in models if model.__class__.__name__ == best_model)

        # Generate future forecast (7 days)
        if best_model == 'ARIMA':
            future_forecast_diff = best_model_obj.predict(n=14)
            future_forecast_diff = future_forecast_diff[7:]
            future_forecast_vals = future_forecast_diff.values().flatten()
            if np.isnan(future_forecast_vals).any():
                raise ValueError(f"{best_model}: Future forecast contains NaN values")
            last_value = series[-1].values().flatten()[0]
            future_forecast = last_value + future_forecast_vals.cumsum()
            future_df = pd.DataFrame({
                'Total': future_forecast,
                'Date': pd.date_range(start=series.time_index[-1] + pd.Timedelta(days=1), periods=7, freq='D')
            })
            future_df = future_df.set_index('Date')
            future_forecast = TimeSeries.from_dataframe(future_df, freq='D')
        else:
            future_forecast = best_model_obj.predict(n=14)
            future_forecast = future_forecast[7:]

        # Convert TimeSeries to DataFrame
        forecast_df = getattr(future_forecast, TS_DF_METHOD)()
        forecast_df = forecast_df.rename(columns={'Total': 'Total_Predicted'})
        forecast_df['Total_Predicted'] = forecast_df['Total_Predicted'].round().astype(int)
        forecast_df['Tawaf'] = (forecast_df['Total_Predicted'] * 0.65).round().astype(int)
        forecast_df['Saei'] = (forecast_df['Total_Predicted'] * 0.25).round().astype(int)
        forecast_df['Other'] = (forecast_df['Total_Predicted'] * 0.10).round().astype(int)

        # Define crowd level labels based on language
        if language == 'Arabic':
            forecast_df['Crowd_Level'] = forecast_df['Total_Predicted'].apply(
                lambda x: 'منخفض' if x < 66069 else 'متوسط' if x < 89883 else 'مرتفع'
            )
            forecast_df['Tawaf_Crowd_Level'] = forecast_df['Tawaf'].apply(
                lambda x: 'منخفض' if x < 42944 else 'متوسط' if x < 58423 else 'مرتفع'
            )
            forecast_df['Saei_Crowd_Level'] = forecast_df['Saei'].apply(
                lambda x: 'منخفض' if x < 16517 else 'متوسط' if x < 22470 else 'مرتفع'
            )
            forecast_df['Other_Crowd_Level'] = forecast_df['Other'].apply(
                lambda x: 'منخفض' if x < 2000 else 'متوسط' if x < 7800 else 'مرتفع'
            )
        else:
            forecast_df['Crowd_Level'] = forecast_df['Total_Predicted'].apply(
                lambda x: 'Low' if x < 66069 else 'Medium' if x < 89883 else 'High'
            )
            forecast_df['Tawaf_Crowd_Level'] = forecast_df['Tawaf'].apply(
                lambda x: 'Low' if x < 42944 else 'Medium' if x < 58423 else 'High'
            )
            forecast_df['Saei_Crowd_Level'] = forecast_df['Saei'].apply(
                lambda x: 'Low' if x < 16517 else 'Medium' if x < 22470 else 'High'
            )
            forecast_df['Other_Crowd_Level'] = forecast_df['Other'].apply(
                lambda x: 'Low' if x < 2000 else 'Medium' if x < 7800 else 'High'
            )

        forecast_df = forecast_df.reset_index()
        forecast_df['Date'] = forecast_df['Date'].dt.date
        return forecast_df[
            ['Date', 'Total_Predicted', 'Tawaf', 'Saei', 'Other',
             'Crowd_Level', 'Tawaf_Crowd_Level', 'Saei_Crowd_Level', 'Other_Crowd_Level']
        ]

    except Exception as e:
        raise Exception(f"Error generating forecast: {str(e)}")
