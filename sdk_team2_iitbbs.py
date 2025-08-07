import pandas as pd
from untrade.client import Client
from pprint import pprint


import numpy as np
import pandas_ta as ta
from hurst import compute_Hc


def process_data(file_path):
        """Calculate all indicators needed for the strategy"""
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])


        df['MA'] = df['close'].rolling(window=20, min_periods=20).mean()
        std = df['close'].rolling(window=20, min_periods=20).std()
        df['Upper_Band'] = df['MA'] + (std * 2.2)
        df['Lower_Band'] = df['MA'] - (std * 2.2)
        df['BBW'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA']

        class KalmanFilter:
            def __init__(self, Q=1e-5, R=0.01):
                self.Q = Q
                self.R = R
                self.P = 1.0
                self.X = None

            def update(self, measurement):
                if self.X is None:
                    self.X = measurement
                    return self.X

                # Prediction update
                self.P += self.Q

                # Measurement update
                K = self.P / (self.P + self.R)
                self.X += K * (measurement - self.X)
                self.P *= (1 - K)

                return self.X

        # --- Kalman Filter ---
        kf = KalmanFilter()
        filtered_prices = pd.Series(index=df.index, dtype=float)

        for i, price in enumerate(df['close'].values):
            filtered_prices.iloc[i] = kf.update(price)

        df['filtered_price'] = filtered_prices

        # --- Heiken-Ashi Candles ---
        df['Heiken_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['Heiken_Open'] = df['open'].copy()  # Initialize with regular open

        for i in range(1, len(df)):
            df.loc[df.index[i], 'Heiken_Open'] = (df['Heiken_Open'].iloc[i-1] + df['Heiken_Close'].iloc[i-1]) / 2

        df['Heiken_High'] = df[['high', 'Heiken_Open', 'Heiken_Close']].max(axis=1)
        df['Heiken_Low'] = df[['low', 'Heiken_Open', 'Heiken_Close']].min(axis=1)

        # --- Moving Averages and Other Indicators ---
        df["EMA20"] = ta.ema(df['close'], length=20)
        df["EMA03"] = ta.ema(df['close'], length=3)
        df["EMA50"] = ta.ema(df['close'], length=50)
        df["EMA12"] = ta.ema(df['close'], length=12)
        df["EMA26"] = ta.ema(df['close'], length=26)

        # Momentum Indicators
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['RSI_smoothed'] = df['RSI'].rolling(5, min_periods=5).mean()
        df['HMA20'] = ta.hma(df['close'], length=20)

        # Volatility & Strength
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['ADX'] = adx['ADX_14']
        df['ADX_MED'] = df['ADX'].rolling(7, min_periods=7).median()
        df['ADX_STD'] = df['ADX'].rolling(7, min_periods=7).std()

        # MACD Components
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal_Line'] = macd['MACDs_12_26_9']

        # Specialized Indicators
        fisher = ta.fisher(high=df['high'], low=df['low'], length=10)
        df['FDI'] = fisher['FISHERT_10_1']
        df['AO'] = ta.ao(df['high'], df['low'])

        # Volume-based Indicators
        df['EFI'] = ta.efi(df['close'], df['volume'], length=13)
        df['VWMA'] = ta.vwma(df['close'], df['volume'], length=20)
        df['CMF'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)

        # --- Hurst Exponent ---
        window = 100
        # df['Hurst'] = np.nan

        # for i in range(window, len(df)):
        #     # Use only past data for Hurst calculation
        #     price_series = df['close'].iloc[i-window:i].values
        #     H, _, _ = compute_Hc(price_series, kind='random_walk')
        #     df.loc[df.index[i], 'Hurst'] = H


        def calculate_hurst(series, window=100):
            hurst = []
            for i in range(window, len(series)):
                H, _, _ = compute_Hc(series.iloc[i-window:i], kind='random_walk')
                hurst.append(H)
            return pd.Series(hurst, index=series.index[window:], name='Hurst')

        df['Hurst'] = calculate_hurst(df['close']).reindex(df.index)

        # --- Derived Indicators ---
        df['ATR_rolling_mean'] = df['ATR'].rolling(50, min_periods=50).mean()
        df['FDI_threshold'] = 1.2 * (df['ATR'] / df['ATR_rolling_mean'])

        # --- Market Regime Classification ---
        df['regime'] = 'TRANSITION'  # Default regime

        for i in range(max(100, window), len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            window_data = df.iloc[i-50:i]

            bull = (
                pd.notna(current['Hurst']) and
                current['Hurst'] > 0.55 and
                current['FDI'] < current['FDI_threshold'] and
                current['ADX'] > (current['ADX_MED'] + current['ADX_STD']) and
                current['EMA20'] > current['EMA50'] and
                prev['EMA20'] > prev['EMA50'] and
                window_data['EMA20'].iloc[-5:].gt(window_data['EMA50'].iloc[-5:]).all()
            )

            bear = (
                pd.notna(current['Hurst']) and
                current['Hurst'] > 0.55 and
                current['FDI'] < current['FDI_threshold'] and
                current['ADX'] > (current['ADX_MED'] + current['ADX_STD']) and
                current['EMA20'] < current['EMA50'] and
                prev['EMA20'] < prev['EMA50'] and
                window_data['EMA20'].iloc[-5:].lt(window_data['EMA50'].iloc[-5:]).all()
            )

            sideways = (
                pd.notna(current['Hurst']) and
                (current['Hurst'] >= 0.4) and
                (current['Hurst'] <= 0.6) and
                (current['FDI'] > current['FDI_threshold']) and
                (current['ADX'] < 18) and
                (current['BBW'] > 0.1) and
                (abs(current['EMA20'] - current['EMA50']) < current['ATR'])
            )

            if bull:
                df.loc[df.index[i], 'regime'] = 'BULL'
            elif bear:
                df.loc[df.index[i], 'regime'] = 'BEAR'
            elif sideways:
                df.loc[df.index[i], 'regime'] = 'SIDEWAYS'

        return df

def strat(df):


    # Initialize columns
        df['trade_type'] = 'HOLD'
        df['Position'] = 0
        df['signals'] = 0

    # Process signals sequentially to maintain position state
        current_position = 0  # Start with no position

        for i in range(1, len(df)):
        # Get relevant bars (current bar's data is not available yet in live trading)
            current = df.iloc[i]  # The most recent completed bar
            prev = df.iloc[i-1]  # The bar before that

        # --- Define Entry and Exit Conditions ---
        # Long entry conditions
            long_cond_1 = (current['regime'] in ['BULL', 'TRANSITION', 'SIDEWAYS'] and
                        current['EMA20'] > current['EMA50'] and
                        current['Heiken_Open'] < current['EMA20'] and
                        current['Heiken_Close'] > current['EMA20'] and
                        # current['MACD'] > current['MACD_Signal_Line'] and
                        current['RSI_smoothed'] > 55 and
                        current['close'] > current['HMA20'] and
                        # current['BBW'] < 0.2 and
                        # current['AO'] > 6 and
                        current['VWMA'] > current['EMA20'] and
                        current['CMF'] > 0.05
                        # current['EFI'] > 75000
                        )
            # long_cond_2 =
                # (current['regime'] in ['SIDEWAYS'] and
                # # current['close'] <= current['Lower_Band'] * 1.1 and
                # # current['RSI_smoothed'] < 40 and
                # current['CMF'] > 0.1
                # # current['BBW'] > 0.15
                # )
            long_cond_3 = (((current['regime'] in ['SIDEWAYS','BULL'] and prev['regime'] in ['TRANSITION','BEAR']) or
                (current['regime'] in ['TRANSITION', 'BULL', 'SIDEWAYS'] and prev['regime'] in ['BEAR'])) and
                current['high'] > prev['high'] and
                current['CMF'] > 0.15 and
                current['RSI_smoothed'] > 55
                )

        # Short entry conditions
            short_cond_1 = (current['regime'] in ['BEAR', 'TRANSITION','SIDEWAYS'] and
                current['EMA20'] < current['EMA50'] and
                current['Heiken_Open'] > current['EMA20'] and
                current['Heiken_Close'] < current['EMA20'] and
                # current['MACD'] > current['MACD_Signal_Line'] and
                current['RSI_smoothed'] < 40 and
                current['close'] < current['HMA20'] and
                # current['BBW'] < 0.2 and
                # current['AO'] < -6 and
                current['VWMA'] < current['EMA20'] and
                current['CMF'] < 0.1 
                # current['EFI'] < -75000
                )
            # short_cond_2 = (current['regime'] in ['SIDEWAYS','TRANSITION'] and
            #     current['close'] >= current['Upper_Band'] * 0.995 and
            #     current['RSI_smoothed'] > 60 and
            #     current['CMF'] < 0 and
            #     current['BBW'] > 0.1)
            short_cond_3 = (((current['regime'] in ['TRANSITION', 'BEAR', 'SIDEWAYS'] and prev['regime'] in ['BULL'])) and
                # (current['regime'] in ['BEAR'] and prev['regime'] in ['TRANSITION','SIDEWAYS'])) and
                # current['high'] > prev['high'] and
                current['CMF'] < 0 and
                current['RSI'] < 35
                )

        # Exit long conditions
            exit_long_cond_1 = (
                current_position == 1 and (
                    # (current['MACD'] < current['MACD_Signal_Line'] and prev['MACD'] > prev['MACD_Signal_Line']) or  # MACD crossover down
                    # (current['AO'] < 0 and prev['AO'] > 0) or  # Awesome Oscillator crossing below zero
                    # (current['CMF'] < -0.05 and prev['CMF'] > -0.05) or  # Money flow turning negative
                    # (current['RSI_smoothed'] < 45 and prev['RSI_smoothed'] > 45) or # RSI falling below 45
                    (current['regime'] == 'BEAR') or
                    (current['EMA20'] < current['EMA50'] and current['Heiken_Close'] < current['close']) or
                    (current['regime'] in ['BEAR','TRANSITION'] and prev['regime'] in ['SIDEWAYS'])
                )
            )
            exit_long_cond_3 = (
                current_position == 1 and (
                    # (current['MACD'] < current['MACD_Signal_Line'] and prev['MACD'] > prev['MACD_Signal_Line']) or  # MACD crossover down
                    # (current['AO'] < 0 and prev['AO'] > 0) or  # Awesome Oscillator crossing below zero
                    # (current['CMF'] < -0.05 and prev['CMF'] > -0.05) or  # Money flow turning negative
                    # (current['RSI_smoothed'] < 45 and prev['RSI_smoothed'] > 45) or # RSI falling below 45
                    (current['regime'] == 'BEAR') or
                    (current['EMA20'] < current['EMA50'] and current['Heiken_Close'] < current['close']) or
                    (current['regime'] in ['BEAR','TRANSITION'] and prev['regime'] in ['SIDEWAYS'])
                )
            )

            # Exit short conditions
            exit_short_cond = (
                current_position == -1 and (
                    current['regime'] == 'BULL' or
                    (current['EMA20'] > current['EMA50'] and current['Heiken_Close'] > current['close']) or
                    (current['regime'] in ['BULL','TRANSITION'] and prev['regime'] in ['SIDEWAYS'])
                )
            )

            # --- Position Logic ---
            new_position = current_position
            trade_type = 'HOLD'
            signal = 0
            lev = 1
            position=100

            # Apply signal logic with safeguards against consecutive same signals
            if current_position == 0:  # No current position
                if long_cond_1 :
                    new_position = 1
                    trade_type = 'LONG'
                    signal = 1
                    lev = 1
                elif long_cond_3:
                    new_position = 1
                    trade_type = 'LONG'
                    signal = 1
                    lev = 1
                    position=50
                elif short_cond_3 or short_cond_1:
                    new_position = -1
                    trade_type = 'SHORT'
                    signal = -1
                    lev = 1
            elif current_position == 1:  # Currently long
                if exit_long_cond_1 or exit_long_cond_3:  # Exit long position
                    new_position = 0
                    trade_type = 'CLOSE'
                    signal = -1
                elif short_cond_3 or short_cond_1:  # Reverse from long to short
                    new_position = -1
                    trade_type = 'REVERSE_LONG'
                    signal = -2
                    lev = 1
                else:
                    # Already in a LONG position, explicitly HOLD
                    new_position = 1
                    trade_type = 'HOLD'
                    signal = 0
            elif current_position == -1:  # Currently short
                if long_cond_1 :  # Reverse from short to long
                    new_position = 1
                    trade_type = 'REVERSE_SHORT'
                    signal = 2
                    lev = 1
                elif long_cond_3:  # Reverse from short to long
                    new_position = 1
                    trade_type = 'REVERSE_SHORT'
                    signal = 2
                    lev = 1
                    position = 50
                elif exit_short_cond:  # Exit short position
                    new_position = 0
                    trade_type = 'CLOSE'
                    signal = 1
                else:
                    # Already in a SHORT position, explicitly HOLD
                    new_position = -1
                    trade_type = 'HOLD'
                    signal = 0

            # Update position, trade type, and signal
            df.loc[df.index[i], 'Position'] = new_position
            df.loc[df.index[i], 'trade_type'] = trade_type
            df.loc[df.index[i], 'signals'] = signal
            df.loc[df.index[i], 'leverage'] = lev
            df.loc[df.index[i], 'position'] = position

            # Update current_position for next iteration
            current_position = new_position

        # Add compatibility columns
        df['position_state'] = df['Position']
        df['leverage'] = df['leverage'].fillna(0)
        df['position'] = df['position'].fillna(100)

        return df


# # Usage
# processed_data = process_data('BTC_2019_2023_1h.csv')
# strategy_results = strat(processed_data)
# cols_to_check = ['open', 'close', 'datetime', 'volume', 'high', 'low']
# strategy_results = strategy_results[ ~(strategy_results[cols_to_check].isin([0, np.nan, pd.NaT]).any(axis=1))]
# strategy_results.to_csv('losho3_1.csv', index=False)
# # Specify the file name
# # file_path = 'test_11.csv'

# # # Download the file
# # files.download(file_path)
# # print('The file is downloaded')


def perform_backtest(csv_file_path):
    """
    Perform backtesting using the untrade SDK.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing historical price data and signals.

    Returns:
    - result (generator): Generator object that yields backtest results.
    """
    # Create an instance of the untrade client
    client = Client()

    # Perform backtest using the provided CSV file path
    result = client.backtest(
        jupyter_id="vijaym",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path# Adjust leverage as needed
    )

    return result


if __name__ == "__main__":
    # Read data from CSV file
    # data = pd.read_csv("BTC_2019_2023_1h.csv")

    # Process data
    res1 = process_data("C:/Users/lenovo/Downloads/BTC_2019_2023_1h.csv")

    # Apply strategy
    res = strat(res1)

    cols_to_check = ['open', 'close', 'datetime', 'volume', 'high', 'low']
    res = res[ ~(res[cols_to_check].isin([0, np.nan, pd.NaT]).any(axis=1))]

    # Save processed data to CSV file
    res.to_csv("processed_data.csv", index=False)

    # Perform backtest on processed data
    csv_file_path = "processed_data.csv"
    backtest_result = perform_backtest(csv_file_path)

    # Get the last value of backtest result
    last_value = None
    for value in backtest_result:
        # print(value)  # Uncomment to see the full backtest result (backtest_result is a generator object)
        last_value = value
    print(last_value)