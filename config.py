# Screen region coordinates (left, top, width, height)

# Region to monitor for the "keep" text
TRIGGER_REGION = (710, 935, 502, 69)

# Region for capturing Celsius temperature
CELSIUS_REGION = (1816, 107, 70, 44)

# Region for capturing weather description
WEATHER_DESC_REGION = (1797, 151, 68, 17)

# Region to monitor for "Keepnet" confirmation text
KEEP_NET_CONFIRM_REGION = (878, 12, 163, 61)

# First click region after pressing 'c'
CLICK_REGION_1 = (409, 171, 227, 183)

# Region for checking/capturing bait information
BAIT_REGION = (599, 350, 243, 142)

# Second click region (conditional based on bait check)
CLICK_REGION_2 = (853, 189, 25, 28)

# Region for capturing map name
MAP_REGION = (725, 312, 116, 18) # Renamed from MAP_CONFIRM_REGION for clarity

# Region for capturing game time
GAME_TIME_REGION = (800, 332, 35, 15)

# Region for capturing fish name
FISH_NAME_REGION = (971, 142, 393, 62)

# Region for capturing fish weight
WEIGHT_REGION = (635, 211, 215, 27)

# --- Other Configuration ---

# Name of the waifu2x executable (assuming it's in the same directory or PATH)
WAIFU2X_EXECUTABLE = "waifu2x-ncnn-vulkan.exe"

# Required screen resolution
REQUIRED_RESOLUTION = (1920, 1080)

# Name of the target game process
GAME_PROCESS_NAME = "rf4_x64.exe"

# Delay in seconds
SHORT_DELAY = 0.2

# CSV Log file names
COMPREHENSIVE_LOG_FILE = "rf4_comprehensive_log.csv"
RAW_OCR_LOG_FILE = "raw_ocr_output.csv"

# Temporary directory for images
TEMP_IMAGE_DIR = "temp_images"

# Discord Webhook URL is now loaded from .env file
