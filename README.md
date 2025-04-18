# RF4 Auto Logger

This Python script automates the process of logging fish catches in the game Russian Fishing 4 (RF4). It monitors a specific screen region for a trigger word ("keep"), performs a sequence of in-game actions (key presses, clicks), captures screenshots of relevant information (fish name, weight, bait, map, time, weather), processes these images using OCR (after upscaling with waifu2x), and logs the extracted data into CSV files.

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
*   **Image Upscaling:** Uses `waifu2x-ncnn-vulkan.exe` to upscale captured images for potentially better OCR accuracy.
*   **OCR:** Extracts text from upscaled images using EasyOCR.
*   **Data Formatting:**
    *   Formats Game Time as HH:MM.
    *   Removes "Bait" prefix from Bait Info.
    *   Cleans leading non-alphabetic characters from Weather Description.
*   **Logging:**
    *   Saves structured data to `rf4_comprehensive_log.csv`.
    *   Saves raw OCR output to `raw_ocr_output.csv`.
*   **Debugging:** Saves upscaled images to the `temp_images/` directory.
*   **Safety Checks:**
    *   Verifies screen resolution is 1920x1080 before starting.
    *   Checks if the target game process (`rf4_x64.exe`) is running before monitoring.

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
*   **waifu2x-ncnn-vulkan:** The executable (`waifu2x-ncnn-vulkan.exe`) and its associated `models` folder must be placed in the same directory as the script. Download it from its official source/repository if needed.
*   **Screen Resolution:** The script is configured for a **1920x1080** screen resolution. Coordinates will need adjustment in `config.py` for other resolutions.
*   **Game:** Russian Fishing 4 (`rf4_x64.exe`).

## Setup

1.  **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Installing PyTorch/EasyOCR can take some time and disk space).*
3.  **Place waifu2x:** Download `waifu2x-ncnn-vulkan` and extract its contents (including `waifu2x-ncnn-vulkan.exe` and the `models` folder) into the project directory.
4.  **(Optional) Configure Regions:** If your game layout differs or you use a different resolution, adjust the coordinates defined in `config.py`.
5.  **(Optional) Update License:** Edit the `LICENSE` file to replace `[yyyy]` and `[name of copyright owner]` with the correct year and your name/entity.

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

*   **`rf4_comprehensive_log.csv`:** Contains the main structured data for each catch. Columns: `Timestamp`, `MapName`, `FishName`, `FishWeight`, `BaitInfo`, `GameTime`, `Temperature`, `WeatherDesc`.
*   **`raw_ocr_output.csv`:** Contains the raw output from EasyOCR for each region captured, useful for debugging OCR issues. Columns: `Timestamp`, `Region`, `RawOCRResult`.
*   **`temp_images/`:** Contains the upscaled images used for OCR. This directory is *not* automatically cleaned up upon script exit.

## Configuration (`config.py`)

*   **Region Tuples:** Define the `(left, top, width, height)` coordinates for all screen capture and interaction areas.
*   **`WAIFU2X_EXECUTABLE`:** Name of the waifu2x executable file.
*   **`REQUIRED_RESOLUTION`:** Expected screen resolution (width, height).
*   **`GAME_PROCESS_NAME`:** Name of the game's executable file.
*   **`SHORT_DELAY`:** Default delay used between some actions.
*   **Log File Names:** Customize the names of the output CSV files.
*   **`TEMP_IMAGE_DIR`:** Name of the directory for storing temporary upscaled images.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
