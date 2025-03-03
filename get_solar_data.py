from io import StringIO
import requests
import pandas as pd

# Your NREL API Key
API_KEY = "dpBD04Vn13sHyLsZXwDIWS2Nuhu1oTewSTHiv6s5"

# Your Email (Required for Large Requests)
EMAIL = "diegobus@stanford.edu"

# Direct CSV Download (Single Point Only)
WKT_POINT = "POINT(-115.3939 33.8214)"  # Desert Sunlight Solar Farm Center
URL_CSV = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"

# Request parameters
PARAMS = {
    "api_key": API_KEY,
    "wkt": WKT_POINT,
    "attributes": "air_temperature,ghi,clearsky_ghi",
    "names": "2023",
    "utc": "true",
    "leap_day": "false",
    "interval": "60",  # Hourly data
    "email": EMAIL,
}


def fetch_and_filter_csv():
    """Fetches CSV data, filters for summer clear days, and saves the result."""
    response = requests.get(URL_CSV, params=PARAMS)

    if response.status_code == 200:
        # Load CSV content into a Pandas dataframe
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, skiprows=2)  # Skip metadata rows

        # Print actual column names retrieved
        print("Column Names in Retrieved CSV:", df.columns.tolist())

        # Standardize column names (strip spaces, convert to lowercase)
        df.columns = df.columns.str.strip().str.lower()

        # Column name mapping (API returned different names)
        column_mapping = {
            "temperature": "air_temperature",
            "ghi": "ghi",
            "clearsky ghi": "clearsky_ghi",
        }

        # Rename columns based on mapping
        df = df.rename(columns=column_mapping)

        # Check if required columns exist
        required_columns = ["air_temperature", "ghi", "clearsky_ghi"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing columns in dataset - {missing_columns}")
            return

        # Convert time components to datetime format
        df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

        # Filter for summer months (June, July, August, September)
        df = df[df["month"].isin([6, 7, 8, 9])]

        # Filter for daytime hours (only when GHI > 50 W/m²)
        df = df[df["ghi"] > 50]

        # Filter for clear days (when GHI is within 5% of clearsky GHI)
        df = df[abs(df["ghi"] - df["clearsky_ghi"]) < 0.05 * df["clearsky_ghi"]]

        # Select relevant columns
        df = df[["time", "air_temperature", "ghi"]]

        # Save filtered dataset to CSV
        df.to_csv("mojave_summer_clear_days.csv", index=False)
        print("Filtered data saved to mojave_summer_clear_days.csv")
    else:
        print(f"Failed to download CSV: {response.status_code}")
        print(f"Response: {response.text}")


# Run the function
fetch_and_filter_csv()
