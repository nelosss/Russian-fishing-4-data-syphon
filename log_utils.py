import pandas as pd
import os
import datetime
import config # Import configuration variables
# import requests # Removed for Discord webhook removal

# Define column headers for the CSV files - UPDATED ORDER
COMPREHENSIVE_COLUMNS = [
    "Timestamp", "MapName", "FishName", "FishWeight",
    "BaitInfo", "GameTime", "Temperature", "WeatherDesc", "WeatherCondition"
]
# Define columns for the new fishing log CSV
FISHING_LOG_COLUMNS = [
    "Timestamp", "MapName", "GameTime", "FishName", "FishWeight",
    "BaitInfo", "Temperature", "WeatherDesc", "WeatherCondition",
    "FishingType", "Coordinates", "Clip", "FloatDepth" # New columns
]
RAW_OCR_COLUMNS = [
    "Timestamp", "Region", "RawOCRResult"
]

def initialize_csv(filename, columns):
    """Creates the CSV file with headers if it doesn't exist."""
    if not os.path.exists(filename):
        try:
            df = pd.DataFrame(columns=columns)
            df.to_csv(filename, index=False, encoding='utf-8-sig') # utf-8-sig for better Excel compatibility
            print(f"Initialized log file: {filename}")
        except Exception as e:
            print(f"Error initializing CSV file {filename}: {e}")

def append_to_comprehensive_log(data):
    """
    Appends a row of data to the comprehensive log CSV.

    Args:
        data (dict): A dictionary where keys match COMPREHENSIVE_COLUMNS.
                     Timestamp should be included automatically.
    """
    filename = config.COMPREHENSIVE_LOG_FILE
    initialize_csv(filename, COMPREHENSIVE_COLUMNS) # Ensure file exists

    try:
        # Add timestamp if not already present
        if "Timestamp" not in data:
            data["Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure all columns are present, fill missing with None or empty string
        row_data = {col: data.get(col, "") for col in COMPREHENSIVE_COLUMNS}

        df = pd.DataFrame([row_data])
        df.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"Appended data to {filename}: {row_data}")
    except Exception as e:
        print(f"Error appending to {filename}: {e}")
        print(f"Data attempted: {data}")


def append_to_raw_ocr_log(region_name, raw_ocr_result):
    """
    Appends raw OCR results for a specific region to the raw OCR log CSV.

    Args:
        region_name (str): Name of the region captured (e.g., "CELSIUS_REGION").
        raw_ocr_result (list): The raw list of tuples from easyocr ([bbox, text, confidence]).
    """
    filename = config.RAW_OCR_LOG_FILE
    initialize_csv(filename, RAW_OCR_COLUMNS) # Ensure file exists

    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Convert raw result list to a string representation for CSV storage
        raw_result_str = str(raw_ocr_result)

        row_data = {
            "Timestamp": timestamp,
            "Region": region_name,
            "RawOCRResult": raw_result_str
        }
        df = pd.DataFrame([row_data])
        df.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"Appended raw OCR data for {region_name} to {filename}")
    except Exception as e:
        print(f"Error appending to {filename}: {e}")
        print(f"Data attempted: Region={region_name}, Result={raw_ocr_result}")


def append_to_csv_log(data, filename="fishing_log.csv"):
    """
    Appends a row of data to the specified fishing log CSV.

    Args:
        data (dict): A dictionary containing the catch data.
                     Keys should ideally match FISHING_LOG_COLUMNS.
        filename (str): The name of the CSV file to append to.
    """
    initialize_csv(filename, FISHING_LOG_COLUMNS) # Ensure file exists with correct headers

    try:
        # Add timestamp if not already present
        if "Timestamp" not in data:
            data["Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure all columns are present, fill missing with empty string
        # Use FISHING_LOG_COLUMNS for the structure
        row_data = {col: data.get(col, "") for col in FISHING_LOG_COLUMNS}

        df = pd.DataFrame([row_data])
        df.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"Appended data to {filename}") # Removed detailed data print for brevity
    except Exception as e:
        print(f"Error appending to {filename}: {e}")
        print(f"Data attempted: {data}")


if __name__ == '__main__':
    # Example Usage (for testing)
    print("Testing log utils...")

    # Test comprehensive log
    test_data_comp = {
        "Temperature": "21C",
        "WeatherDesc": "Cloudy",
        "MapName": "Old Burg",
        "GameTime": "15:30",
        "FishName": "Common Carp",
        "FishWeight": "2.5 kg",
        "BaitInfo": "Bread"
    }
    print("\nAppending test data to comprehensive log...")
    append_to_comprehensive_log(test_data_comp) # Test original log

    # Test new fishing log
    test_data_fishing = {
        "MapName": "Winding Rivulet",
        "GameTime": "08:15",
        "FishName": "Ide",
        "FishWeight": "1.1 kg",
        "BaitInfo": "Mayfly Larva",
        "Temperature": "15C",
        "WeatherDesc": "Sunny",
        "WeatherCondition": "sunny",
        "FishingType": "Float Fishing",
        "Coordinates": "H7",
        "Clip": "", # Empty for float
        "FloatDepth": "1.2"
    }
    print("\nAppending test data to fishing log...")
    append_to_csv_log(test_data_fishing, filename="test_fishing_log.csv") # Use a test filename

    # Test raw OCR log
    test_data_raw = [([10, 10, 50, 20], 'Test Text', 0.95)]
    print("\nAppending test data to raw OCR log...")
    append_to_raw_ocr_log("TEST_REGION", test_data_raw)

    # Test Discord sending (requires valid URL in config.py)
    # Test Discord sending removed
    # print("\nAttempting to send test data to Discord...")
    # send_to_discord(test_data_comp)

    print("\nLog utils test finished. Check CSV files.")


# --- Discord Webhook Function Removed ---
