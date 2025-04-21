# --- Screen Resolution ---
REQUIRED_RESOLUTION = (1920, 1080)

# --- Game Process ---
GAME_PROCESS_NAME = "rf4_x64.exe" # Adjust if your executable name is different

# --- File Paths ---
COMPREHENSIVE_LOG_FILE = "rf4_log.csv"
RAW_OCR_LOG_FILE = "rf4_raw_ocr_log.txt"
TEMP_IMAGE_DIR = "temp_images"
WAIFU2X_EXECUTABLE = "waifu2x-ncnn-vulkan.exe" # Assumes it's in the same dir or PATH

# --- Region Coordinates (X, Y, Width, Height) ---
# IMPORTANT: Adjust these based on your screen setup and game resolution (1920x1080 assumed)
TRIGGER_REGION = (710, 938, 241, 67)      # Area to check for "keep" text
KEEP_NET_CONFIRM_REGION = (878, 12, 163, 61) # Area to check for "Keepnet" text
CLICK_REGION_1 = (409, 171, 227, 183)       # First click area after 'c'
CLICK_REGION_2 = (853, 189, 25, 28)       # Second click area (if no bait text)

# Data Capture Regions
CELSIUS_REGION = (1824, 112, 61, 34)        # Temperature (e.g., "15°C")
WEATHER_DESC_REGION = (1707, 142, 213, 36)  # Weather description (e.g., "Cloudy")
MAP_REGION = (725, 312, 116, 18)          # Map name (e.g., "Old Burg")
GAME_TIME_REGION = (800, 332, 35, 15)     # In-game time (e.g., "15:30")
FISH_NAME_REGION = (971, 142, 393, 62)     # Fish name (after keepnet confirm)
WEIGHT_REGION = (746, 211, 100, 27)      # Fish weight (e.g., "1.234 kg") - Updated Region
BAIT_REGION = (599, 350, 243, 142)       # Bait info area (also used for "Bait" text check)
WEATHER_SYMBOL_REGION = (1730, 15, 172, 95) # Area where weather condition icons appear

# --- Delays (in seconds) ---
SHORT_DELAY = 0.2  # General short delay between actions
LONG_DELAY = 1.0   # Longer delay if needed

# --- Image Processing ---
UPSCALE_MODEL = "models-upconv_7_photo" # Options: models-cunet, models-upconv_7_anime_style_art_rgb, models-upconv_7_photo
SYMBOL_CONFIDENCE = 0.6 # Confidence threshold for symbol detection (wind/water in text AND weather condition icons) - Lowered from 0.7
WEATHER_CONDITION_SYMBOLS = { # Mapping for weather condition icons
    "sunny.png": "sunny",
    "thunder.png": "thunder",
    "cloudy.png": "cloudy",
    "rainy.png": "rainy",
    "sun_with_clouds.png": "partly cloudy",
    "noncloudy.png": "clear sky" # Added new condition
    # Add more symbols here if needed, ensure image files exist
}

# --- OCR Settings ---
# EasyOCR Language Code (e.g., 'en' for English)
OCR_LANGUAGE = 'en'
# Allowlist characters for specific regions (set to None to disable)
# Example: OCR_ALLOWLISTS = {"GameTime": "0123456789:", "FishWeight": "0123456789.kg "}
OCR_ALLOWLISTS = {
    "FishWeight": "0123456789.gk", # Allow digits, dot, g, k for weight
    "GameTime": "0123456789.:" # Allow digits, dot, and colon for time
}

# --- Discord Webhook (Optional) ---
# DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL_HERE" # Replace with your actual webhook URL
DISCORD_WEBHOOK_URL = None # Set to None to disable Discord integration

# --- Overlay Settings ---
OVERLAY_BG_COLOR = 'gray10'
OVERLAY_TEXT_COLOR = 'white'
OVERLAY_FONT_SIZE = 10
OVERLAY_ALPHA = 0.7 # Transparency (0.0 fully transparent, 1.0 fully opaque)
OVERLAY_UPDATE_INTERVAL = 100 # Milliseconds between overlay updates
