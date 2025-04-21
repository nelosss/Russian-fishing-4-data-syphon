# RF4 Auto Logger

This Python script automates the process of logging fish catches in the game Russian Fishing 4 (RF4). It monitors a specific screen region for a trigger word ("keep"), performs a sequence of in-game actions (key presses, clicks), captures screenshots of relevant information (fish name, weight, bait, map, time, weather), processes these images using OCR (after upscaling with waifu2x), detects weather condition icons, and logs the extracted data into CSV files.

## Features

*   **Screen Monitoring:** Watches a defined region for the "keep" trigger word.
*   **Automated Actions:** Simulates spacebar press, 'c' key press, mouse clicks, and Esc key presses.
*   **Conditional Logic:** Checks for "Keepnet" confirmation and presence of "Bait" text to adjust actions.
*   **Screenshot Capture:** Captures specific regions for:
    *   Temperature
    *   Weather Description
    *   Map Name
    *   Game Time
    *   Fish Name
    *   Fish Weight
    *   Bait Information
    *   Weather Condition Icon
*   **Image Upscaling:** Uses `waifu2x-ncnn-vulkan.exe` to upscale captured images for potentially better OCR accuracy.
*   **OCR:** Extracts text from upscaled images using EasyOCR.
*   **Weather Condition Symbol Detection:** Identifies weather icons (sunny, cloudy, rainy, etc.) in a dedicated screen region using template matching.
*   **Data Formatting & Parsing:**
    *   Formats Game Time as HH:MM.
    *   Removes "Bait" prefix from Bait Info.
    *   Robust parsing for Weather Description (keywords, direction, speed with range check). Includes detection of wind/water symbols within the text region.
    *   Robust parsing for Temperature (filters values outside -1 to 38°C, prioritizes decimals).
    *   Robust parsing for Fish Weight (handles 'g' vs 'kg', corrects likely OCR errors based on game formatting rules).
*   **Logging:**
    *   Saves structured data to `rf4_log.csv`. Includes a new `WeatherCondition` column.
    *   Saves raw OCR output to `rf4_raw_ocr_log.txt`.
*   **Debugging:**
    *   Saves upscaled images used for OCR to the `temp_images/` directory.
    *   Saves the captured image of the weather symbol region to `temp_images/` for debugging detection issues (can be disabled).
*   **Safety Checks:**
    *   Verifies screen resolution matches `REQUIRED_RESOLUTION` in `config.py`.
*   Checks if the target game process (`rf4_x64.exe`) is running before monitoring.

## Simple How-To Guide (For Beginners)

Here's a basic guide to get the logger running:

1.  **Get the Code:**
    *   Go to the main page of this project on GitHub.
    *   Click the green "Code" button, then choose "Download ZIP".
    *   Extract the downloaded ZIP file to a folder on your computer (e.g., on your Desktop).

2.  **Install Python:**
    *   If you don't have Python, download and install it from [python.org](https://www.python.org/). Make sure to check the box that says "Add Python to PATH" during installation.

3.  **Install Required Tools:**
    *   Open the Command Prompt (search for "cmd" in the Windows Start menu).
    *   Navigate to the folder where you extracted the code using the `cd` command. For example, if it's on your Desktop in a folder named `RF4-Auto-Logger-main`, type:
        ```bash
        cd Desktop\RF4-Auto-Logger-main
        ```
    *   Install the necessary Python packages by typing this command and pressing Enter:
        ```bash
        pip install -r requirements.txt
        ```
        *(This might take a few minutes).*

4.  **Get Image Upscaler (waifu2x):**
    *   This tool helps the script read text from screenshots better. You need to download `waifu2x-ncnn-vulkan`. Search for it online (look for releases on GitHub).
    *   Download the zip file, extract it, and copy the `waifu2x-ncnn-vulkan.exe` file AND the entire `models` folder into the same folder where you put the script code (from Step 1).

5.  **Prepare Weather Symbol Images:**
    *   You need small image files for each weather condition icon (e.g., `sunny.png`, `cloudy.png`, `rainy.png`, `thunder.png`, `sun_with_clouds.png`).
    *   **Important:** These images should be screenshots of the *actual white icons* shown in the game at your resolution (1920x1080). Crop them tightly around the icon itself.
    *   Place these image files in the same folder as the `rf4_autologger.py` script.
    *   Make sure the filenames match the keys in the `WEATHER_CONDITION_SYMBOLS` dictionary in `config.py`.
    *   (If you have black-line templates, the `invert_templates.py` script can attempt to invert them, but using actual screenshots is recommended).

6.  **Check Screen Resolution & Configure:**
    *   Ensure your screen resolution is **1920x1080**.
    *   Open `config.py` in a text editor. Verify the coordinates for `TRIGGER_REGION`, `WEATHER_SYMBOL_REGION`, and other capture areas match your game layout. Adjust if necessary.
    *   Check the `WEATHER_CONDITION_SYMBOLS` dictionary in `config.py` to ensure filenames and desired text output are correct.

7.  **Run the Logger:**
    *   Make sure Russian Fishing 4 is running.
    *   Go back to the Command Prompt window (still in the script's folder).
    *   Type the following command and press Enter:
        ```bash
        python rf4_autologger.py
        ```
    *   The script will ask if you want to show region overlays (type 'y' or 'n').
    *   It will then watch your game screen. When you catch a fish and the "keep" button appears, it should automatically log the details.

8.  **Stop the Logger:**
    *   Go back to the Command Prompt window where the script is running.
    *   Press `Ctrl` + `C` (hold the Control key and press C).

9.  **Find Your Logs:**
    *   Look inside the script's folder. You'll find `rf4_log.csv` containing your logged fishing data. You can open this with programs like Microsoft Excel or Google Sheets.
    *   `rf4_raw_ocr_log.txt` contains detailed OCR output for troubleshooting.
    *   The `temp_images` folder contains screenshots used for processing (including the weather symbol region for debugging).

## Requirements

*   **Python 3.x**
*   **Required Python Packages:** See `requirements.txt` (install via `pip install -r requirements.txt`). Main dependencies include:
    *   `easyocr` (and its dependencies like `torch`, `torchvision`)
    *   `opencv-python`
    *   `mss`
    *   `pandas`
    *   `numpy`
    *   `PyAutoGUI`
    *   `psutil`
    *   `Pillow` (for image inversion script)
*   **waifu2x-ncnn-vulkan:** The executable (`waifu2x-ncnn-vulkan.exe`) and its associated `models` folder must be placed in the same directory as the script. Download it from its official source/repository if needed.
*   **Weather Symbol Images:** Image files (e.g., `sunny.png`, `cloudy.png`) corresponding to the in-game weather icons, placed in the same directory as the script. **These should ideally be screenshots of the white in-game icons.** (The `invert_templates.py` script can help invert black-line templates).
*   **Screen Resolution:** The script is configured by default for **1920x1080**. Coordinates will need adjustment in `config.py` for other resolutions.
*   **Game:** Russian Fishing 4 (`rf4_x64.exe`).

## Setup

1.  **Clone Repository or Download ZIP:** Get the code files.
    ```bash
    # Example using git
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Installing PyTorch/EasyOCR can take some time and disk space).*
3.  **Place waifu2x:** Download `waifu2x-ncnn-vulkan` and extract its contents (including `waifu2x-ncnn-vulkan.exe` and the `models` folder) into the project directory.
4.  **Prepare Weather Symbol Images:** Create tightly cropped screenshots of each white in-game weather icon (sunny, cloudy, etc.) and save them as `.png` files (e.g., `sunny.png`, `cloudy.png`) in the project directory. Ensure filenames match the keys in `WEATHER_CONDITION_SYMBOLS` in `config.py`. (Alternatively, use `invert_templates.py` on black-line images, but screenshots are preferred).
5.  **Configure `config.py`:** Open `config.py` and adjust region coordinates (`TRIGGER_REGION`, `WEATHER_SYMBOL_REGION`, etc.) to match your screen layout at 1920x1080 resolution. Verify `WEATHER_CONDITION_SYMBOLS` dictionary.

## Usage

1.  Ensure Russian Fishing 4 is running.
2.  Navigate to the project directory in your terminal.
3.  Run the main script:
    ```bash
    python rf4_autologger.py
    ```
4.  The script will start monitoring the screen. When the "keep" text appears in the `TRIGGER_REGION`, it will perform the automated sequence.
5.  Press `Ctrl+C` in the terminal to stop the script.

## Log Files

*   **`rf4_log.csv`:** Contains the main structured data for each catch. Columns: `Timestamp`, `MapName`, `FishName`, `FishWeight`, `BaitInfo`, `GameTime`, `Temperature`, `WeatherDesc`, `WeatherCondition`.
*   **`rf4_raw_ocr_log.txt`:** Contains the raw output from EasyOCR for each region captured, useful for debugging OCR issues. Columns: `Timestamp`, `Region`, `RawOCRResult`.
*   **`temp_images/`:** Contains the upscaled images used for OCR and the captured weather symbol region image (if debugging enabled). This directory is *not* automatically cleaned up upon script exit.

## Configuration (`config.py`)

*   **Region Tuples:** Define the `(left, top, width, height)` coordinates for all screen capture and interaction areas (e.g., `TRIGGER_REGION`, `WEATHER_SYMBOL_REGION`).
*   **`WAIFU2X_EXECUTABLE`:** Name of the waifu2x executable file.
*   **`UPSCALE_MODEL`:** Specifies the model used by waifu2x (e.g., `models-upconv_7_photo`).
*   **`SYMBOL_CONFIDENCE`:** Confidence threshold (0.0 to 1.0) for image matching (wind/water symbols in text and weather condition icons).
*   **`WEATHER_CONDITION_SYMBOLS`:** Dictionary mapping weather icon filenames (e.g., `sunny.png`) to the text to be logged (e.g., `"sunny"`).
*   **`REQUIRED_RESOLUTION`:** Expected screen resolution (width, height).
*   **`GAME_PROCESS_NAME`:** Name of the game's executable file.
*   **`SHORT_DELAY`:** Default delay used between some actions.
*   **Log File Names:** Customize the names of the output log files (`COMPREHENSIVE_LOG_FILE`, `RAW_OCR_LOG_FILE`).
*   **`TEMP_IMAGE_DIR`:** Name of the directory for storing temporary images.
*   **`OCR_LANGUAGE`:** Language code for EasyOCR (e.g., `'en'`).
*   **`OCR_ALLOWLISTS`:** Dictionary to restrict OCR characters for specific regions (e.g., `{"FishWeight": "0123456789.gk"}`).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
