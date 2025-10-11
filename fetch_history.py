import pandas as pd
from openalgo import api
from datetime import datetime
from dotenv import load_dotenv
import os
import argparse

# 🔁 OpenAlgo Python Bot is running.

def fetch_history(symbol, exchange, interval, start_date, end_date, output_csv=None):
    """Fetch historical data from OpenAlgo API"""
    # Load environment variables
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    API_HOST = os.getenv("OPENALGO_API_HOST")

    # Initialize OpenAlgo client
    client = api(api_key=API_KEY, host=API_HOST)

    if output_csv is None:
        output_csv = f"{symbol}_history.csv"

    # Fetch Historical Data
    df = client.history(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start_date=start_date,
        end_date=end_date
    )

    # Convert index to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Save to CSV
    df.to_csv(output_csv)
    print(f"✅ Historical data saved to {output_csv}")
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch historical data from OpenAlgo')
    parser.add_argument('--symbol', type=str, default="NIFTY28OCT2525200PE", help='Trading symbol')
    parser.add_argument('--exchange', type=str, default="NFO", help='Exchange (NFO, NSE, etc.)')
    parser.add_argument('--interval', type=str, default="5m", help='Time interval (1m, 5m, 15m, 1h, D)')
    parser.add_argument('--start_date', type=str, default="2025-09-01", help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default="2025-10-06", help='End date (YYYY-MM-DD)')
    parser.add_argument('--output_csv', type=str, default=None, help='Output CSV filename')

    args = parser.parse_args()

    fetch_history(
        symbol=args.symbol,
        exchange=args.exchange,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        output_csv=args.output_csv
    )
