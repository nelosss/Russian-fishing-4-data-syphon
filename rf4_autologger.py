import time
import os
import datetime
import uuid
import shutil
import sys
import re # Import regex module
import queue
import threading
import csv # Added for reading CSV
import keyboard # Added for hotkey support
# from dotenv import load_dotenv # Removed .env loading

# Third-party libraries
import pyautogui
import psutil
import cv2 # Added for OpenCV template matching
import numpy as np # Added for OpenCV operations

# Project modules
import config
import input_utils
import image_utils
import screen_utils
import log_utils
from overlay_utils import OverlayManager # Import the manager

# --- Globals for Background Processing ---
processing_queue = queue.Queue()
# overlay_update_queue is now managed by OverlayManager instance
processing_results = {} # Stores {original_path: (success_bool, result_text, raw_ocr)}
results_lock = threading.Lock()
stop_worker_event = threading.Event() # To signal the worker thread to stop
stop_main_event = threading.Event()   # To signal the main logger thread to stop

# --- Globals for Fishing Setup ---
current_fishing_setup = {} # Stores {'FishingType': '...', 'Coordinates': '...', 'Clip': '...', 'FloatDepth': '...'}
fishing_setup_lock = threading.Lock() # To protect access to current_fishing_setup

# --- Region Colors ---
# Define colors for the region frames
REGION_COLORS = {
    "Trigger": "cyan",
    "KeepnetConfirm": "magenta",
    "Click1": "orange",
    "Click2": "orange",
    "BaitRegionCheck": "pink", # For the check_text_in_region call for bait
    # Colors for the main captured regions
    "Temperature": "lightblue",
    "WeatherDesc": "lightgreen",
    "MapName": "red",
    "GameTime": "yellow",
    "FishName": "white",
    "FishWeight": "green",
    "BaitInfo": "purple" # Same region as BaitRegionCheck, but different context
}

# --- Helper Functions ---

def normalize_text(text):
    """Normalizes text for weight extraction: lowercase, comma->dot, unit fixes, strip chars."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = t.replace(',', '.')
    # Fix common unit misreads
    t = t.replace('k9', 'kg').replace('kq', 'kg').replace('kgs', 'kg')
    # Fix common OCR number errors
    t = t.replace('o', '0').replace('l', '1')
    # Keep only relevant characters: numbers, letters (for kg/g), dot, space
    t = re.sub(r'[^0-9a-z\.\s]', '', t)
    # Collapse multiple spaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_weights(text):
    """Extracts weights (kg or g) from normalized text, returns list of weights in grams (int)."""
    print(f"[Weight Extractor] Received Raw Text: '{text}'") # Debug: Show raw input
    clean = normalize_text(text)
    print(f"[Weight Extractor] Normalized Text: '{clean}'") # Debug: Show normalized text
    # Regex: number (up to 6 digits.up to 3 decimals) followed by optional space and kg or g
    # Made space optional with \s* instead of \s+ or \s
    matches = re.findall(r'(\d{1,6}(?:\.\d{1,3})?)\s*(kg|g)\b', clean)
    results = []
    print(f"[Weight Extractor] Regex Matches: {matches}") # Debug print
    if not matches:
        print(f"[Weight Extractor] No matches found with primary regex.")
        # Fallback: Try finding just a number if no unit match
        num_matches = re.findall(r'(\d{1,6}(?:\.\d{1,3})?)', clean)
        if num_matches:
            print(f"[Weight Extractor] Fallback: Found numbers without units: {num_matches}")
            # Try to parse the last found number, assuming grams as a last resort
            try:
                last_num_str = num_matches[-1]
                val = float(last_num_str)
                # Basic heuristic: if it has a decimal or is large, maybe it was kg?
                # This is risky, but better than nothing if primary regex fails.
                # Let's just assume grams for simplicity unless it's clearly decimal.
                if '.' in last_num_str:
                     # If it has a decimal, it's likely kg that lost its unit. Convert.
                     print(f"[Weight Extractor] Fallback: Assuming decimal number '{last_num_str}' is kg -> {int(val * 1000)}g")
                     results.append(int(val * 1000))
                elif val >= 1000: # Large number, likely grams already
                     print(f"[Weight Extractor] Fallback: Assuming large number '{last_num_str}' is g -> {int(val)}g")
                     results.append(int(val))
                else: # Small number, assume grams
                     print(f"[Weight Extractor] Fallback: Assuming small number '{last_num_str}' is g -> {int(val)}g")
                     results.append(int(val))
            except ValueError:
                print(f"[Weight Extractor] Fallback: Error converting value '{last_num_str}' to float.")
        else:
            print(f"[Weight Extractor] Fallback: No numbers found either.")


    for val_str, unit in matches: # This loop only runs if primary regex found matches
        print(f"[Weight Extractor] Processing Match: Value='{val_str}', Unit='{unit}'") # Debug: Show match being processed
        try:
            val = float(val_str)
            grams = 0
            # Manual conversion to grams with heuristic for misread 'kg'
            if unit == 'kg':
                if val >= 1000: # Heuristic: If number is large, assume OCR misread 'kg' for 'g'
                    print(f"[Weight Extractor] Unit is 'kg' but value {val} >= 1000. Assuming grams.")
                    grams = int(val)
                else: # Reasonable kg value, convert
                    grams = int(val * 1000)
            else: # Unit is 'g'
                grams = int(val)
            results.append(grams)
        except ValueError:
            print(f"[Weight Extractor] Error converting value '{val_str}' to float.")
            continue # Skip this match if conversion fails
    print(f"[Weight Extractor] Extracted grams: {results}") # Debug print
    return results

def load_fish_max_weights(filepath="fish_weights.csv"):
    """Loads fish max weights from a CSV file."""
    weights = {}
    if not os.path.exists(filepath):
        print(f"Warning: Max weight file not found: {filepath}")
        return weights # Return empty dict if file not found

    try:
        with open(filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            next(reader) # Skip header row 1
            next(reader) # Skip header row 2
            for row in reader:
                if len(row) >= 2:
                    fish_name = row[0].strip()
                    weight_str = row[1].strip().lower()
                    # Clean the weight string (remove ' g', spaces)
                    cleaned_weight_str = weight_str.replace('g', '').replace(' ', '')
                    try:
                        weight_grams = int(cleaned_weight_str)
                        # Treat the placeholder large number as infinity/None
                        if weight_grams >= 9999999:
                             weights[fish_name] = float('inf') # Use infinity for no practical limit
                             print(f"Loaded '{fish_name}' with no practical max weight.")
                        else:
                             weights[fish_name] = weight_grams
                             # print(f"Loaded '{fish_name}' with max weight: {weight_grams}g") # Too verbose for normal use
                    except ValueError:
                        print(f"Warning: Could not parse weight for '{fish_name}': '{weight_str}'")
                else:
                    print(f"Warning: Skipping invalid row in {filepath}: {row}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"Error reading max weight file {filepath}: {e}")

    print(f"Loaded {len(weights)} fish max weights from {filepath}.")
    return weights

def parse_captured_weight_to_grams(weight_str):
    """Parses OCR'd weight string ('123 g', '1.234 kg') into grams (int). Returns None on failure."""
    if not isinstance(weight_str, str):
        return None # Handle non-string input

    weight_str = weight_str.strip().lower()
    weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g)\b', weight_str)

    if weight_match:
        weight_num_str = weight_match.group(1)
        unit = weight_match.group(2)
        try:
            weight_float = float(weight_num_str)
            if unit == 'kg':
                # Convert kg to grams
                return int(weight_float * 1000)
            elif unit == 'g':
                # Return grams directly (ensure it's an integer)
                return int(weight_float)
        except ValueError:
            print(f"Error converting parsed weight '{weight_num_str}' to float.")
            return None # Conversion failed
    else:
        # Attempt to parse if only a number is present (assume grams)
        try:
            # Remove any trailing non-digits (like potential 'g' without space)
            cleaned_num_str = re.sub(r'[^\d.]+$', '', weight_str).strip()
            if cleaned_num_str:
                 weight_float = float(cleaned_num_str)
                 # Heuristic: if it has a decimal or is >= 1000, assume it was kg misread as g
                 if '.' in cleaned_num_str or weight_float >= 1000:
                     print(f"Assuming '{weight_str}' was misread kg, converting {weight_float} to grams.")
                     return int(weight_float) # Treat as grams value directly if large or decimal
                 else:
                     return int(weight_float) # Assume grams if small integer
            else:
                 return None # No parsable number found
        except ValueError:
             print(f"Error parsing weight string '{weight_str}' as grams.")
             return None # Parsing failed

    return None # Default return if no match or error


def is_process_running(process_name):
    """Check if a process with the given name is currently running."""
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == process_name:
            return True
    return False

def clean_temp_files(directory=config.TEMP_IMAGE_DIR):
    """Removes the temporary image directory and its contents."""
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            print(f"Cleaned up temporary directory: {directory}")
        except Exception as e:
            print(f"Error cleaning up temporary directory {directory}: {e}")

def create_temp_dir():
    """Creates the temporary directory if it doesn't exist."""
    if not os.path.exists(config.TEMP_IMAGE_DIR):
        os.makedirs(config.TEMP_IMAGE_DIR)
        print(f"Created temporary directory: {config.TEMP_IMAGE_DIR}")

# This function remains synchronous as it runs before the main async part
def process_and_log_capture(manager, region_name, region_coords, log_data, raw_log_func): # Pass manager
    """Captures, upscales, OCRs a region, and logs the results (synchronously)."""
    timestamp_uuid = f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    base_filename = os.path.join(config.TEMP_IMAGE_DIR, f"{region_name}_{timestamp_uuid}")
    original_img_path = f"{base_filename}_orig.png"
    upscaled_img_path = f"{base_filename}_upscaled.png"

    print(f"\n--- Processing Region (Sync): {region_name} ---")
    manager.update_queue.put(f"Processing {region_name}...") # Update overlay status

    # 1. Capture Original
    img_array = screen_utils.capture_region(region_coords, filename=original_img_path)
    if img_array is None:
        print(f"Failed to capture {region_name}.")
        log_data[region_name] = "Capture Failed" # Log failure
        manager.update_queue.put(f"{region_name}: Capture Failed")
        return # Skip rest of processing for this region

    detected_symbol_prefix = "" # Store "wind " or "water " if detected
    # --- Symbol Detection (WeatherDesc only) ---
    if region_name == "WeatherDesc":
        print(f"Checking for weather symbols in {original_img_path}...")
        wind_symbol_path = 'wind.png'
        water_symbol_path = 'water.png'
        symbol_found = False # Keep track locally if we found one

        try:
            # Check for wind symbol
            if os.path.exists(wind_symbol_path):
                location = pyautogui.locate(wind_symbol_path, original_img_path, confidence=config.SYMBOL_CONFIDENCE)
                if location:
                    print("Wind symbol detected.")
                    detected_symbol_prefix = "wind " # Set prefix with space
                    symbol_found = True
            else:
                print(f"Warning: {wind_symbol_path} not found, cannot check for wind symbol.")

            # Check for water symbol if wind not found
            if not symbol_found and os.path.exists(water_symbol_path):
                location = pyautogui.locate(water_symbol_path, original_img_path, confidence=config.SYMBOL_CONFIDENCE)
                if location:
                    print("Water symbol detected.")
                    detected_symbol_prefix = "water " # Set prefix with space
                    symbol_found = True
            elif not symbol_found:
                 print(f"Warning: {water_symbol_path} not found, cannot check for water symbol.")

            if not symbol_found:
                 print("Weather symbols (wind/water) not detected.")
            # No early return here, always proceed to OCR

        except pyautogui.ImageNotFoundException:
            print("Weather symbols (wind/water) not found using pyautogui.locate.")
        except Exception as e:
            print(f"Error during symbol detection for {region_name}: {e}.")
            # Fall through to OCR

    # --- End Symbol Detection ---

    # 2. Upscale (Always runs now)
    print(f"Upscaling {region_name} image...")
    upscale_success, upscale_error_msg = image_utils.upscale_image(original_img_path, upscaled_img_path)
    if not upscale_success:
        error_details = f"Reason: {upscale_error_msg}" if upscale_error_msg else "Unknown reason."
        print(f"Failed to upscale {region_name}. {error_details} Attempting OCR on original.")
        manager.update_queue.put(f"{region_name}: Upscale Failed ({error_details[:50]}...), OCR on original...") # Truncate long errors for overlay
        # Fallback: try OCR on the original image if upscale fails
        allowlist_for_ocr = config.OCR_ALLOWLISTS.get(region_name) # Use config allowlist
        print(f"[Sync Fallback] Using allowlist for {region_name}: '{allowlist_for_ocr}'")
        ocr_texts, raw_ocr = screen_utils.perform_ocr(original_img_path, allowlist=allowlist_for_ocr)
        log_data[region_name] = "Upscale Failed | OCR: " + ", ".join(ocr_texts) if ocr_texts else "Upscale Failed | OCR Failed"
        raw_log_func(region_name, raw_ocr)
        # Update overlay with fallback results
        overlay_lines = [f"{region_name} (Original):"]
        if raw_ocr:
            for _, text, conf in raw_ocr:
                overlay_lines.append(f"  '{text}' (Conf: {conf:.2f})")
        else:
            overlay_lines.append("  OCR Failed")
        manager.update_queue.put("\n".join(overlay_lines))
        # Delete original image after failed upscale + fallback OCR attempt
        if os.path.exists(original_img_path):
            try:
                os.remove(original_img_path)
                print(f"Deleted original image: {original_img_path}")
            except Exception as e:
                print(f"Error deleting original image {original_img_path}: {e}")
        return

    # 3. Perform OCR on Upscaled Image
    print(f"Performing OCR on upscaled {region_name} image...")
    manager.update_queue.put(f"{region_name}: Upscaled, Performing OCR...")
    allowlist_for_ocr = config.OCR_ALLOWLISTS.get(region_name) # Use config allowlist
    print(f"[Sync] Using allowlist for {region_name}: '{allowlist_for_ocr}'")
    ocr_texts, raw_ocr = screen_utils.perform_ocr(upscaled_img_path, allowlist=allowlist_for_ocr)
    overlay_lines = [f"{region_name} (Upscaled):"] # Start overlay text
    if not ocr_texts:
        print(f"OCR failed for upscaled {region_name}.")
        log_data[region_name] = "OCR Failed"
        overlay_lines.append("  OCR Failed")
    else:
        # Join multiple text results if necessary
        joined_text = ", ".join(ocr_texts).replace('\n', ' ').strip()

        # Apply formatting rules (copied from worker for consistency)
        if region_name == "BaitInfo":
            parts = [part.strip() for part in joined_text.split(',')]
            filtered_parts = [part for part in parts if part.lower() != "bait"]
            processed_text = ", ".join(filtered_parts)
            print(f"Processed BaitInfo OCR: {processed_text}")
        elif region_name == "GameTime" and len(joined_text) == 4 and joined_text.isdigit():
            processed_text = f"{joined_text[:2]}:{joined_text[2:]}"
            print(f"Formatted GameTime OCR: {processed_text}")
        elif region_name == "WeatherDesc":
            print(f"Raw WeatherDesc OCR: {joined_text}")
            # Attempt to parse WeatherDesc using regex extraction
            processed_text = "Parse Failed" # Default if parsing fails
            lower_text = joined_text.lower()

            # Check for keywords first
            if "calm" in lower_text:
                processed_text = "calm"
            elif "normal" in lower_text:
                 processed_text = "normal"
            elif "reduced" in lower_text:
                 processed_text = "reduced"
            else:
                # Try to extract direction and speed
                # More robust direction matching (N, NE, E, SE, S, SW, W, NW)
                direction_match = re.search(r'\b(N|NE|E|SE|S|SW|W|NW)\b', joined_text, re.IGNORECASE)
                # Try to find a number (integer or decimal) - more flexible
                speed_match = re.search(r'(\d+(\.\d+)?)', joined_text)

                if direction_match and speed_match:
                    direction = direction_match.group(1).upper()
                    speed_str = speed_match.group(1)
                    try:
                        speed_float = float(speed_str)
                        # Add validity check for speed (e.g., 0 to 15 m/s)
                        if 0.0 <= speed_float <= 15.0:
                             # Reconstruct in the desired format
                            processed_text = f"{direction} {speed_float} m/s" # Use float for consistent formatting
                            print(f"Parsed WeatherDesc: Direction='{direction}', Speed='{speed_float}'")
                        else:
                            print(f"Ignoring implausible wind speed: {speed_float}. Falling back.")
                            processed_text = "Parse Failed" # Mark as failed if speed is out of range
                    except ValueError:
                         print(f"Could not convert speed '{speed_str}' to float. Falling back.")
                         processed_text = "Parse Failed" # Mark as failed if conversion fails

                # If parsing failed (no match, bad speed, or conversion error), fall back
                if processed_text == "Parse Failed":
                    print("Could not parse valid direction/speed, using basic cleaning as fallback.")
                    match = re.search(r'[a-zA-Z]', joined_text)
                    processed_text = joined_text[match.start():].strip() if match else joined_text

            print(f"Processed WeatherDesc: {processed_text}")

        elif region_name == "Temperature":
            print(f"Raw Temperature OCR: {joined_text}")
            processed_text = "Parse Failed" # Default
            temperature_value = None
            valid_numbers = []

            # Find all potential numbers (integer or decimal)
            all_numbers_found = re.findall(r'(\d+(\.\d+)?)', joined_text) # Returns list of tuples like [('18.1', '.1'), ('0', '')]

            # Filter for plausible temperatures (-1.0 to 38.0)
            for num_tuple in all_numbers_found:
                num_str = num_tuple[0]
                try:
                    num_float = float(num_str)
                    # Use the user-specified range check
                    if -1.0 <= num_float <= 38.0:
                        valid_numbers.append(num_str)
                    else:
                        print(f"Ignoring temperature value outside range (-1 to 38): {num_float}")
                except ValueError:
                    print(f"Could not convert found number '{num_str}' to float.")
                    continue # Skip if conversion fails

            if valid_numbers:
                found_decimal = False
                # Prioritize valid numbers with a decimal point
                for num_str in valid_numbers:
                    if '.' in num_str:
                        temperature_value = num_str
                        found_decimal = True
                        print(f"Found valid decimal temperature: {temperature_value}")
                        break # Use the first valid decimal number found

                # If no valid decimal number was found, use the first valid number extracted
                if not found_decimal:
                    temperature_value = valid_numbers[0] # Use the first valid number
                    print(f"No valid decimal temp found, using first valid number: {temperature_value}")

                processed_text = f"{temperature_value}°C"
            else:
                # Fallback if no *valid* numbers are found
                print("Could not parse any valid temperature value.")
                processed_text = "Parse Failed" # Set to Parse Failed instead of raw text

            # Only print if parsing didn't fail
            if processed_text != "Parse Failed":
                print(f"Processed Temperature: {processed_text}")
        else:
             processed_text = joined_text

        # Prepend symbol prefix if detected
        final_text = detected_symbol_prefix + processed_text
        log_data[region_name] = final_text
        print(f"OCR Result ({region_name}): {final_text}") # Log final text
        # Add successful OCR results to overlay text
        overlay_lines.append(f"  Symbol: {detected_symbol_prefix.strip() if detected_symbol_prefix else 'None'}") # Show detected symbol
        for _, text, conf in raw_ocr:
            overlay_lines.append(f"  '{text}' (Conf: {conf:.2f})")

    # 4. Log Raw OCR Data
    raw_log_func(region_name, raw_ocr)
    # Update overlay with final results for this region
    manager.update_queue.put("\n".join(overlay_lines))

    # 5. Delete Original Image (if upscale was successful)
    if os.path.exists(original_img_path):
        try:
            os.remove(original_img_path)
            print(f"Deleted original image: {original_img_path}")
        except Exception as e:
            print(f"Error deleting original image {original_img_path}: {e}")

    print(f"--- Finished Processing Region (Sync): {region_name} ---")


def detect_weather_condition(region_coords, symbols_dict, confidence):
    """Detects the first matching weather symbol in a region."""
    print(f"\n--- Detecting Weather Condition Symbol ---")
    print(f"Region: {region_coords}")
    condition = "None" # Default if no symbol found
    best_match_score = -1.0 # Initialize best score
    match_threshold = 0.4 # Confidence threshold for OpenCV match (Lowered from 0.5)
    temp_capture_path = os.path.join(config.TEMP_IMAGE_DIR, f"weather_capture_{uuid.uuid4().hex[:6]}.png")

    # 1. Capture the symbol region
    img_array = screen_utils.capture_region(region_coords, filename=temp_capture_path)
    if img_array is None:
        print("Failed to capture weather symbol region.")
        return condition # Return default

    try:
        # Load the captured region in grayscale
        region_img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        print(f"Loaded captured region: {temp_capture_path}")

        # 2. Loop through template symbols and check for match using OpenCV
        for template_file, symbol_text in symbols_dict.items():
            template_path = os.path.join(os.getcwd(), template_file) # Ensure full path
            if not os.path.exists(template_path):
                print(f"Warning: Symbol template file not found: {template_path}. Skipping.")
                continue

            try:
                # Load the template image in grayscale
                template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template_img is None:
                    print(f"Error: Failed to load template image: {template_path}. Skipping.")
                    continue

                # Check if template is smaller than the region image
                if template_img.shape[0] > region_img_gray.shape[0] or template_img.shape[1] > region_img_gray.shape[1]:
                     print(f"Warning: Template {template_file} is larger than the captured region. Skipping.")
                     continue

                # Perform template matching
                # TM_CCOEFF_NORMED: Normalised correlation coefficient - good for finding best match
                result = cv2.matchTemplate(region_img_gray, template_img, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # Find the best match score (max_val)

                print(f"  Matching {template_file}: Max score = {max_val:.4f}")

                # Check if this match is better than previous ones and above threshold
                if max_val > best_match_score and max_val >= match_threshold:
                    best_match_score = max_val
                    condition = symbol_text
                    print(f"    New best match found: {symbol_text} (Score: {best_match_score:.4f})")

            except Exception as inner_e:
                 print(f"Error processing template {template_file}: {inner_e}")

    except Exception as e:
        print(f"An error occurred during OpenCV weather symbol detection: {e}")
    finally:
        # Keep temporary capture image for debugging if needed
        if os.path.exists(temp_capture_path):
             print(f"Keeping temporary weather capture image: {temp_capture_path}")
            # try:
            #     os.remove(temp_capture_path)
            # except Exception as e:
            #     print(f"Error deleting temp weather capture image {temp_capture_path}: {e}")
        else:
             print(f"Temporary weather capture image not found (capture might have failed): {temp_capture_path}")

    print(f"Final Detected Condition: {condition} (Best Score: {best_match_score:.4f})")
    print(f"--- Finished Detecting Weather Condition Symbol (OpenCV) ---")
    return condition


# --- Fishing Setup Functions ---

def prompt_fishing_setup():
    """Prompts user for fishing type and details, returns setup dict."""
    print("\n--- Configure Fishing Setup ---")
    print("1. Bottom Fishing")
    print("2. Spin Fishing")
    print("3. Float Fishing")
    fishing_type = None
    while fishing_type is None:
        choice = input("Select Fishing Type (1, 2, or 3): ").strip()
        if choice == '1':
            fishing_type = "Bottom Fishing"
        elif choice == '2':
            fishing_type = "Spin Fishing"
        elif choice == '3':
            fishing_type = "Float Fishing"
        else:
            print("Invalid choice.")

    setup = {"FishingType": fishing_type}
    coords = input(f"Enter Coordinates for {fishing_type} (e.g., G5): ").strip()
    setup["Coordinates"] = coords if coords else "N/A"

    if fishing_type == "Bottom Fishing":
        clip = input(f"Enter Clip for {fishing_type} (e.g., 35): ").strip()
        setup["Clip"] = clip if clip else "N/A"
        setup["FloatDepth"] = ""
    elif fishing_type == "Float Fishing":
        depth = input(f"Enter Float Depth for {fishing_type} (e.g., 1.5): ").strip()
        setup["FloatDepth"] = depth if depth else "N/A"
        setup["Clip"] = ""
    else: # Spin Fishing
        setup["Clip"] = ""
        setup["FloatDepth"] = ""

    print("--- Fishing Setup Complete ---")
    print(f" Type: {setup.get('FishingType', 'N/A')}")
    print(f" Coords: {setup.get('Coordinates', 'N/A')}")
    if setup.get('Clip'): print(f" Clip: {setup['Clip']}")
    if setup.get('FloatDepth'): print(f" Depth: {setup['FloatDepth']}")
    print("-----------------------------")
    return setup

def trigger_setup_update():
    """Called by hotkey to re-prompt for fishing setup."""
    global current_fishing_setup # Declare intent to modify global
    print("\n--- Ctrl+7 Detected: Updating Fishing Setup ---")
    # It's generally discouraged to do blocking input in a hotkey callback,
    # but for this specific use case (user explicitly wants to pause and reconfigure),
    # it might be acceptable. Be aware this will block the hotkey listener thread.
    try:
        new_setup = prompt_fishing_setup()
        with fishing_setup_lock: # Protect global variable access
            current_fishing_setup = new_setup
        print("--- Fishing Setup Updated ---")
        # Optionally, update overlay here if manager is accessible
        # manager.update_queue.put("Fishing setup updated via hotkey.")
    except Exception as e:
        print(f"Error during hotkey setup update: {e}")

# --- Main Execution ---

def main(manager, stop_event): # Removed fishing_type parameter
    """Main logging loop."""
    # Removed .env loading
    # Max weights are no longer loaded here as the check was removed.
    # global FISH_MAX_WEIGHTS # Removed
    # FISH_MAX_WEIGHTS = load_fish_max_weights() # Removed

    print("--- RF4 Auto Logger Starting ---")
    print(f"Monitoring Trigger Region: {config.TRIGGER_REGION}")
    print(f"Required Resolution: {config.REQUIRED_RESOLUTION}")
    print(f"Target Process: {config.GAME_PROCESS_NAME}")
    print(f"Log Files: {config.COMPREHENSIVE_LOG_FILE}, {config.RAW_OCR_LOG_FILE}")
    print("-" * 30)
    manager.update_queue.put("Logger Started. Waiting for game...")

    # 1. Initial Checks
    # Check screen resolution
    current_resolution = pyautogui.size()
    if current_resolution != config.REQUIRED_RESOLUTION:
        print(f"Error: Incorrect screen resolution. Expected {config.REQUIRED_RESOLUTION}, got {current_resolution}.")
        manager.update_queue.put(f"ERROR: Bad Resolution ({current_resolution})")
        # Don't sys.exit, allow main thread to handle shutdown
        return

    # Check if EasyOCR initialized successfully
    if screen_utils.reader is None:
         print("Error: EasyOCR failed to initialize. Cannot continue.")
         manager.update_queue.put("ERROR: EasyOCR Failed")
         return

    # Check if waifu2x executable exists
    if not os.path.exists(config.WAIFU2X_EXECUTABLE):
        print(f"Error: Waifu2x executable not found at '{config.WAIFU2X_EXECUTABLE}'. Please ensure it's in the script directory or PATH.")
        manager.update_queue.put("ERROR: waifu2x not found")
        # Don't exit here, image_utils will handle it per-call, but warn user.

    # 2. Create Temp Directory
    create_temp_dir()

    # 3. Main Loop
    print("\nStarting main loop. Press Ctrl+C in console (or close overlay) to exit.")
    processing_catch = False # Flag to prevent re-triggering
    try:
        while not stop_event.is_set(): # Check stop event
            # Check if game process is running
            if not is_process_running(config.GAME_PROCESS_NAME):
                status_msg = f"Waiting for {config.GAME_PROCESS_NAME}..."
                print(status_msg + " (Check every 5s)")
                manager.update_queue.put(status_msg)
                time.sleep(5)
                continue # Skip the rest of the loop iteration

            # Check for "keep" only if not already processing a catch
            if not processing_catch:
                manager.update_queue.put("Monitoring for 'keep'...")
                if screen_utils.check_text_in_region(config.TRIGGER_REGION, "keep"):
                    print("\n>>> 'keep' detected! Starting action sequence. <<<")
                    manager.update_queue.put("'keep' detected! Starting...")
                    processing_catch = True # Set flag to prevent immediate re-trigger

                    # --- Action Sequence ---
                    input_utils.delay(config.SHORT_DELAY) # Add delay *before* space press for timing
                    input_utils.press_key('space')
                    input_utils.delay(config.SHORT_DELAY) # Keep delay *after* space

                    # Prepare data storage for this catch
                    catch_data = {}
                    raw_ocr_data_func = lambda name, raw: log_utils.append_to_raw_ocr_log(name, raw)

                    # Capture/Process Temp & Weather (before 'c') - Synchronously
                    # These already update the overlay via process_and_log_capture
                    process_and_log_capture(manager, "Temperature", config.CELSIUS_REGION, catch_data, raw_ocr_data_func)
                    process_and_log_capture(manager, "WeatherDesc", config.WEATHER_DESC_REGION, catch_data, raw_ocr_data_func)

                    input_utils.delay(config.SHORT_DELAY) # Specified 0.2s delay
                    input_utils.press_key('c')

                    # Wait for "Keepnet" confirmation (with timeout)
                    print("Waiting for 'Keepnet' confirmation...")
                    manager.update_queue.put("Waiting for 'Keepnet'...")
                    keepnet_found = False
                    timeout = time.time() + 10 # 10 second timeout
                    while time.time() < timeout:
                        if screen_utils.check_text_in_region(config.KEEP_NET_CONFIRM_REGION, "Keepnet"):
                            print("'Keepnet' confirmed!")
                            manager.update_queue.put("'Keepnet' confirmed!")
                            keepnet_found = True
                            break
                        time.sleep(0.5) # Check every 0.5 seconds
                        if stop_event.is_set(): break # Allow early exit

                    if stop_event.is_set(): break # Exit loop if stopped

                    if keepnet_found:
                        # --- Post-Keepnet Actions ---

                        # --- Get CURRENT Fishing Setup ---
                        with fishing_setup_lock: # Safely read global setup
                            active_fishing_setup = current_fishing_setup.copy()
                        catch_data.update(active_fishing_setup) # Add current setup details
                        print(f"Applying current setup: {active_fishing_setup.get('FishingType')}, Coords: {active_fishing_setup.get('Coordinates')}")
                        manager.update_queue.put(f"Using setup: {active_fishing_setup.get('FishingType')}")

                        # --- Post-Keepnet Actions (Clicks) ---
                        # Moved clicks after getting details to avoid interrupting input
                        input_utils.click_within_region(config.CLICK_REGION_1)
                        input_utils.delay(config.SHORT_DELAY) # Delay after click

                        # Check for "Bait" text in BAIT_REGION
                        if not screen_utils.check_text_in_region(config.BAIT_REGION, "Bait"):
                            print("No 'Bait' text found, clicking second region.")
                            manager.update_queue.put("No 'Bait', clicking region 2...")
                            input_utils.click_within_region(config.CLICK_REGION_2)
                            input_utils.delay(config.SHORT_DELAY) # Delay after click
                        else:
                            print("'Bait' text found, skipping second click.")
                            manager.update_queue.put("'Bait' found, skipping click 2.")


                        # --- Capture remaining info and queue for processing ---
                        images_to_process_details = [] # Store details needed for queuing
                        tasks_for_this_catch = [] # Store original_paths to wait for results
                        regions_to_capture = {
                            "MapName": config.MAP_REGION,
                            "GameTime": config.GAME_TIME_REGION,
                            "FishName": config.FISH_NAME_REGION,
                            "FishWeight": config.WEIGHT_REGION,
                            "BaitInfo": config.BAIT_REGION
                        }

                        print("\nCapturing remaining regions and queuing for processing...")
                        manager.update_queue.put("Capturing remaining regions...")
                        for region_name, region_coords in regions_to_capture.items():
                            if stop_event.is_set(): break # Allow early exit

                            timestamp_uuid = f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
                            base_filename = os.path.join(config.TEMP_IMAGE_DIR, f"{region_name}_{timestamp_uuid}")
                            original_img_path = f"{base_filename}_orig.png"
                            upscaled_img_path = f"{base_filename}_upscaled.png"

                            print(f"Capturing {region_name}...")
                            img_array = screen_utils.capture_region(region_coords, filename=original_img_path)
                            if img_array is None:
                                print(f"Failed to capture {region_name}. Logging failure.")
                                catch_data[region_name] = "Capture Failed" # Log capture failure directly
                                manager.update_queue.put(f"Capture Failed: {region_name}")
                            else:
                                # Add task details for the worker queue
                                task_details = (original_img_path, upscaled_img_path, region_name)
                                processing_queue.put(task_details)
                                tasks_for_this_catch.append(original_img_path) # Track this task
                                images_to_process_details.append({"region_name": region_name, "original_path": original_img_path}) # Keep for result mapping
                                print(f"Queued task for {region_name} ({os.path.basename(original_img_path)})")
                                manager.update_queue.put(f"Queued: {region_name}")
                            time.sleep(0.05) # Tiny delay between captures

                        if stop_event.is_set(): break # Allow early exit

                        # --- Press Esc twice IMMEDIATELY after captures ---
                        print("Pressing Esc twice...")
                        input_utils.press_key('esc', presses=2, interval=config.SHORT_DELAY)

                        # --- Wait for and retrieve results from the worker ---
                        print("\nWaiting for background processing results...")
                        manager.update_queue.put("Waiting for processing...")
                        any_processing_failed = False # Flag for upscale OR OCR failure OR timeout
                        results_retrieved_count = 0
                        max_wait_time = 30 # seconds for all tasks
                        start_wait_time = time.time()

                        processed_task_keys = set() # Keep track of keys processed to avoid duplicates

                        while results_retrieved_count < len(tasks_for_this_catch) and (time.time() - start_wait_time) < max_wait_time:
                            if stop_event.is_set(): break # Allow early exit

                            found_new_result = False
                            with results_lock:
                                # Check for results for tasks related to *this specific catch*
                                current_result_keys = set(processing_results.keys())
                                tasks_to_check = set(tasks_for_this_catch)

                                # Find which tasks for this catch have results ready and haven't been processed yet
                                ready_tasks = (tasks_to_check & current_result_keys) - processed_task_keys

                                if ready_tasks:
                                    for original_path in ready_tasks:
                                        success_flag, processed_text, raw_ocr = processing_results[original_path]

                                        # Find the corresponding region_name
                                        region_name = "Unknown"
                                        for item in images_to_process_details:
                                            if item["original_path"] == original_path:
                                                region_name = item["region_name"]
                                                break

                                        print(f"Retrieved result for {region_name} (Success: {success_flag})")
                                        catch_data[region_name] = processed_text # Store the final text
                                        raw_ocr_data_func(region_name, raw_ocr) # Log raw OCR data

                                        if not success_flag:
                                            any_processing_failed = True # Mark failure if upscale OR OCR failed in worker

                                        processed_task_keys.add(original_path) # Mark as processed
                                        results_retrieved_count += 1
                                        found_new_result = True
                                        # Overlay is updated by the worker thread now

                            if not found_new_result and results_retrieved_count < len(tasks_for_this_catch):
                                # Only sleep if no new results were found in this check cycle
                                time.sleep(0.2) # Wait a bit before checking again

                        if stop_event.is_set(): break # Allow early exit

                        # After waiting loop (or timeout)
                        if results_retrieved_count < len(tasks_for_this_catch):
                            print(f"Warning: Timed out waiting for all processing results ({results_retrieved_count}/{len(tasks_for_this_catch)} received).")
                            manager.update_queue.put("WARNING: Processing Timeout")
                            # Mark as failed if not all results came back
                            any_processing_failed = True
                            # Log placeholder for missing results
                            with results_lock: # Need lock to safely iterate processing_results keys
                                processed_paths_in_results = set(processing_results.keys()) & set(tasks_for_this_catch)
                            missing_tasks = set(tasks_for_this_catch) - processed_paths_in_results
                            for original_path in missing_tasks:
                                 region_name = "Unknown"
                                 for item in images_to_process_details:
                                     if item["original_path"] == original_path:
                                         region_name = item["region_name"]
                                         break
                                 catch_data[region_name] = "Processing Timeout"
                                 print(f"Marking {region_name} as 'Processing Timeout'.")


                        # --- Clean up results for this catch from the global dictionary ---
                        print("Cleaning up processed results...")
                        with results_lock:
                            # Use list comprehension to avoid modifying dict while iterating in some Python versions
                            keys_to_delete = [key for key in tasks_for_this_catch if key in processing_results]
                            for key in keys_to_delete:
                                del processing_results[key]

                        # --- Detect Weather Condition Symbol ---
                        detected_condition = detect_weather_condition(
                            config.WEATHER_SYMBOL_REGION,
                            config.WEATHER_CONDITION_SYMBOLS,
                            config.SYMBOL_CONFIDENCE
                        )
                        catch_data["WeatherCondition"] = detected_condition # Add to data for logging

                        # --- Log comprehensive data (CSV and TXT) ---
                        # Always attempt to log whatever data was collected, even if processing failed
                        print("Logging catch data to fishing_log.csv...")
                        manager.update_queue.put("Logging catch data...")
                        log_utils.append_to_csv_log(catch_data, filename="fishing_log.csv") # Log to CSV
                        # log_utils.append_to_comprehensive_log(catch_data) # REMOVED call to old log function
                        if any_processing_failed:
                             print("Note: Some processing steps failed or timed out; logged data may be incomplete.")
                             manager.update_queue.put("Logged (possibly incomplete).")
                        else:
                             print("Logging complete.")
                             manager.update_queue.put("Logging complete.")
                        # --- Discord sending removed ---
                        # log_utils.send_to_discord(catch_data)

                        print("\n>>> Action sequence complete (processing done after Esc). Resuming monitoring. <<<")
                        # Add a delay AFTER completing the sequence to prevent immediate re-triggering
                        print("Adding post-sequence delay...")
                        manager.update_queue.put("Sequence complete. Delaying...")
                        # Use a loop for the delay to check stop_event periodically
                        delay_end_time = time.time() + 2.0 # Reduced delay from 5.0 to 2.0
                        while time.time() < delay_end_time:
                            if stop_event.is_set(): break
                            time.sleep(0.1)
                        processing_catch = False # Reset flag after delay


                    else: # Keepnet not found
                        print("Timeout waiting for 'Keepnet'. Aborting sequence.")
                        manager.update_queue.put("Timeout waiting for 'Keepnet'.")
                        # Optional: Press Esc to back out if stuck
                        input_utils.press_key('esc')
                        processing_catch = False # Reset flag if sequence aborted

            # Pause before next check to reduce CPU usage, check stop event
            loop_delay_end_time = time.time() + 0.5
            while time.time() < loop_delay_end_time:
                 if stop_event.is_set(): break
                 time.sleep(0.1)


    except KeyboardInterrupt:
        print("\nCtrl+C detected in main logger thread. Signaling stop.")
        stop_event.set() # Signal stop to self and potentially others
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {e}")
        manager.update_queue.put(f"ERROR in main loop: {e}")
        stop_event.set() # Signal stop on error
    finally:
        # Clean up temporary files on exit - COMMENTED OUT TO KEEP UPSCALED IMAGES
        # clean_temp_files()
        print("--- RF4 Auto Logger Main Thread Stopped ---")
        manager.update_queue.put("Logger thread stopped.")
        # Ensure stop event is set for other threads
        stop_event.set()


# --- Background Image Processor ---

def image_processor_worker(manager): # Accept manager
    """Worker thread function to process images from the queue."""
    print("[Worker] Image processing worker thread started.")
    while not stop_worker_event.is_set():
        processed_something = False # Flag to check if we processed anything in this cycle
        overlay_lines = [] # Store lines for overlay update for the current task
        try:
            # Wait for a task, with a timeout to allow checking the stop event
            task = processing_queue.get(timeout=1)
            if task is None: # Sentinel value to stop
                break

            original_path, upscaled_path, region_name = task
            print(f"[Worker] Processing task for {region_name} ({os.path.basename(original_path)})")
            overlay_lines.append(f"Processing: {region_name}...")
            manager.update_queue.put("\n".join(overlay_lines)) # Initial status update

            success_flag = False # Represents success of BOTH upscale AND OCR
            processed_text = "Processing Error"
            raw_ocr = {} # Default empty raw OCR
            upscale_success = False # Track upscale success separately for cleanup logic

            thresholded_path = None # Path for thresholded image, if created
            try:
                # 1. Upscale
                scale = 4 if region_name == "FishWeight" else 2 # Use 4x for weight, 2x otherwise
                print(f"[Worker] Upscaling {region_name} image ({scale}x)...")
                # Use model from config
                upscale_success, upscale_error_msg = image_utils.upscale_image(original_path, upscaled_path, scale_factor=scale, model=config.UPSCALE_MODEL)

                if not upscale_success:
                    error_details = f"Reason: {upscale_error_msg}" if upscale_error_msg else "Unknown reason."
                    print(f"[Worker] Failed to upscale {region_name}. {error_details} Attempting OCR on original.")
                    overlay_lines[-1] = f"{region_name}: Upscale Failed, OCR Orig..." # Update last status
                    overlay_lines.append(f"  Error: {upscale_error_msg}") # Add specific error to overlay
                    manager.update_queue.put("\n".join(overlay_lines))
                    # Determine allowlist for fallback OCR
                    allowlist_for_ocr = config.OCR_ALLOWLISTS.get(region_name) # Use config allowlist
                    print(f"[Worker] Using allowlist for fallback OCR on {region_name}: '{allowlist_for_ocr}'")
                    # Fallback OCR on original image
                    ocr_texts, raw_ocr = screen_utils.perform_ocr(original_path, allowlist=allowlist_for_ocr)
                    processed_text = "Upscale Failed | OCR: " + ", ".join(ocr_texts) if ocr_texts else "Upscale Failed | OCR Failed"
                    # Add fallback OCR results to overlay
                    overlay_lines.append(f"  Result (Orig):")
                    if raw_ocr:
                         for _, text, conf in raw_ocr:
                             overlay_lines.append(f"    '{text}' ({conf:.2f})")
                    else:
                         overlay_lines.append("    OCR Failed")
                    # success_flag remains False
                else:
                    # --- Optional Thresholding for FishWeight (DISABLED) ---
                    ocr_input_path = upscaled_path # Default to using the upscaled image
                    # if region_name == "FishWeight":
                    #     print(f"[Worker] Applying thresholding to 4x upscaled {region_name} image...")
                    #     try:
                    #         # Add check if file exists before reading
                    #         if not os.path.exists(upscaled_path):
                    #             raise FileNotFoundError(f"Upscaled image not found at {upscaled_path}")
                    #         img = cv2.imread(upscaled_path)
                    #         if img is None:
                    #             raise ValueError(f"Failed to load upscaled image for thresholding from {upscaled_path}")
                    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #         # Apply adaptive thresholding - parameters might need tuning
                    #         # Increased block size, adjusted C
                    #         thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    #                                            cv2.THRESH_BINARY_INV, 15, 5) # Block size 15, C 5
                    #
                    #         # Save thresholded image temporarily
                    #         thresholded_path = upscaled_path.replace('_upscaled.png', '_thresholded.png')
                    #         write_success = cv2.imwrite(thresholded_path, thresh_img)
                    #         if not write_success:
                    #              raise IOError(f"Failed to write thresholded image to {thresholded_path}")
                    #         print(f"[Worker] Saved thresholded image to: {thresholded_path}")
                    #         ocr_input_path = thresholded_path # Use thresholded image for OCR
                    #     except Exception as thresh_e:
                    #         print(f"[Worker] Error during thresholding for {region_name}: {thresh_e}. Falling back to upscaled image.")
                    #         thresholded_path = None # Ensure path is None if thresholding failed
                    #         ocr_input_path = upscaled_path # Fallback to original upscaled
                    # --- End Thresholding ---

                    # 2. Perform OCR
                    print(f"[Worker] Performing OCR on {os.path.basename(ocr_input_path)} for {region_name}...")
                    overlay_lines[-1] = f"{region_name}: Upscaled ({scale}x)" # Update status
                    if thresholded_path:
                         overlay_lines[-1] += ", Thresholded"
                    overlay_lines[-1] += ", OCR..."
                    manager.update_queue.put("\n".join(overlay_lines))

                    # Determine allowlist for OCR from config
                    allowlist_for_ocr = config.OCR_ALLOWLISTS.get(region_name)
                    print(f"[Worker] Using allowlist for {region_name}: '{allowlist_for_ocr}'")

                    ocr_texts, raw_ocr = screen_utils.perform_ocr(ocr_input_path, allowlist=allowlist_for_ocr)
                    if not ocr_texts:
                        print(f"[Worker] OCR failed for {os.path.basename(ocr_input_path)} ({region_name}).")
                        processed_text = "OCR Failed"
                        overlay_lines.append("  Result (Upscaled): OCR Failed")
                        # success_flag remains False (as OCR failed)
                    else:
                        # 3. Join and apply formatting rules
                        joined_text = ", ".join(ocr_texts).replace('\n', ' ').strip()
                        # Apply specific formatting rules based on region_name
                        if region_name == "BaitInfo":
                            parts = [part.strip() for part in joined_text.split(',')]
                            filtered_parts = [part for part in parts if part.lower() != "bait"]
                            processed_text = ", ".join(filtered_parts)
                            # If filtering removed everything (likely just "Bait"), keep original text
                            if not processed_text.strip():
                                print("[Worker] BaitInfo filtering resulted in empty string. Using original OCR text.")
                                processed_text = joined_text # Fallback to original joined text
                        elif region_name == "GameTime":
                            # Format HHMM to HH:MM only if exactly 4 digits, otherwise use cleaned text
                            cleaned_time = re.sub(r'[^\d.]', '', joined_text) # Keep only digits and dot
                            if len(cleaned_time) == 4 and cleaned_time.isdigit():
                                processed_text = f"{cleaned_time[:2]}:{cleaned_time[2:]}"
                            else:
                                processed_text = cleaned_time # Use the cleaned result (e.g., "21.35")
                        elif region_name == "FishWeight":
                            print(f"[Worker] Raw FishWeight OCR: {joined_text}")
                            # Replace commas with dots first
                            text_with_dots = joined_text.replace(',', '.')
                            # Fix common OCR number errors (O->0, l->1)
                            text_corrected = text_with_dots.replace('O', '0').replace('o', '0').replace('l', '1')
                            # Extract only digits and dots
                            processed_text = re.sub(r'[^0-9.]', '', text_corrected)
                            print(f"[Worker] Extracted digits/dots for FishWeight: {processed_text}")
                            # Handle empty string case after filtering
                            if not processed_text:
                                print("[Worker] FishWeight extraction resulted in empty string. Marking as 'Parse Failed'.")
                                processed_text = "Parse Failed"

                        elif region_name == "WeatherDesc":
                            # This logic is now handled synchronously in process_and_log_capture
                            # Keep a basic fallback here just in case, though it shouldn't be hit often
                            match = re.search(r'[a-zA-Z]', joined_text)
                            processed_text = joined_text[match.start():].strip() if match else joined_text
                            print(f"[Worker] Fallback WeatherDesc processing: {processed_text}")
                        elif region_name == "Temperature": # Temp is usually processed sync, but keep logic
                             # This logic is now handled synchronously in process_and_log_capture
                             # Keep a basic fallback here
                             if re.search(r'\d', joined_text) and not joined_text.upper().endswith('C'):
                                cleaned_temp = re.sub(r'[^\d.]+$', '', joined_text).strip()
                                processed_text = cleaned_temp + "°C" if cleaned_temp else joined_text
                             else:
                                processed_text = joined_text
                        else:
                             processed_text = joined_text

                        print(f"[Worker] OCR Result ({region_name}): {processed_text}")
                        success_flag = True # Upscale and OCR succeeded
                        # Add successful OCR results to overlay
                        overlay_lines.append(f"  Result (Upscaled):")
                        for _, text, conf in raw_ocr:
                            overlay_lines.append(f"    '{text}' ({conf:.2f})")


            except Exception as e:
                print(f"[Worker] Error processing {region_name} ({os.path.basename(original_path)}): {e}")
                processed_text = f"Processing Exception: {e}"
                overlay_lines.append(f"  ERROR: {e}")
                success_flag = False
            finally:
                # Store result regardless of success
                with results_lock:
                    processing_results[original_path] = (success_flag, processed_text, raw_ocr)
                print(f"[Worker] Stored result for {os.path.basename(original_path)} (Success: {success_flag})")
                overlay_lines[0] = f"Done: {region_name} (Success: {success_flag})" # Update first line status
                manager.update_queue.put("\n".join(overlay_lines)) # Send final update for this task
                processed_something = True

                # Clean up original image file
                if os.path.exists(original_path):
                    try:
                        os.remove(original_path)
                        # print(f"[Worker] Deleted original image: {original_path}") # Maybe too verbose
                    except Exception as e:
                        print(f"[Worker] Error deleting original image {original_path}: {e}")

                # Keep upscaled images
                # Clean up thresholded image if it was created and still exists
                if thresholded_path and os.path.exists(thresholded_path):
                    try:
                        os.remove(thresholded_path)
                        print(f"[Worker] Deleted temporary thresholded image: {thresholded_path}")
                    except Exception as e:
                        print(f"[Worker] Error deleting thresholded image {thresholded_path}: {e}")
                # Keep original upscaled image (upscaled_path)
                # if not upscale_success and os.path.exists(upscaled_path): # Logic to delete failed upscale if desired
                #     try:
                #         os.remove(upscaled_path)
                #     except Exception as e:
                #         print(f"[Worker] Error deleting failed upscaled image {upscaled_path}: {e}")

                # Signal task completion *inside* the finally block
                processing_queue.task_done()
        # Remove extraneous line that caused SyntaxError
        # 354,069 / 1,048.576K tokens used (34%)
        except queue.Empty:
            # Queue is empty, just loop again and check stop_worker_event
            # Optionally clear overlay if nothing was processed for a while?
            # if not processed_something:
            #    manager.update_queue.put("Worker Idle...")
            continue
        except Exception as e:
            # Catch unexpected errors in the worker loop itself
            print(f"[Worker] Unexpected error in worker loop: {e}")
            manager.update_queue.put(f"WORKER ERROR: {e}")
            time.sleep(1) # Avoid busy-looping on error

    print("[Worker] Image processing worker thread stopped.")


if __name__ == "__main__":
    manager = None # Initialize manager variable
    main_logger_thread = None
    worker_thread = None
    # selected_fishing_type = None # Removed

    try:
        # --- Get Initial Fishing Setup ---
        # Prompt for the initial setup and store it globally
        initial_setup = prompt_fishing_setup()
        with fishing_setup_lock:
            current_fishing_setup = initial_setup
        # --- End Initial Fishing Setup ---

        # --- Setup Hotkey ---
        try:
            # Note: Using 'ctrl+f7' as 'ctrl+7' might conflict depending on keyboard layout/OS
            # Using 'suppress=True' means the original Ctrl+F7 action (if any) won't happen.
            # The callback function `trigger_setup_update` will be executed.
            keyboard.add_hotkey('ctrl+7', trigger_setup_update, suppress=True)
            print("Registered hotkey Ctrl+7 to update fishing setup.")
            print("Press Ctrl+7 anytime to re-configure.")
        except Exception as e:
            print(f"Warning: Failed to register hotkey 'ctrl+7'. Error: {e}")
            print("Hotkey functionality will be disabled.")
        # --- End Setup Hotkey ---

        # Create Overlay Manager
        # Ask user about showing region frames
        show_frames_input = input("Show region frame overlays? (y/n): ").lower().strip()
        show_region_frames = show_frames_input == 'y'

        # Create Overlay Manager
        manager = OverlayManager()

        # Create the text overlay part
        manager.create_text_overlay()

        # Create region frame overlays from config ONLY if requested
        if show_region_frames:
            print("Creating region frame overlays...")
            all_regions = {
                "Trigger": config.TRIGGER_REGION,
                "KeepnetConfirm": config.KEEP_NET_CONFIRM_REGION,
                "Click1": config.CLICK_REGION_1,
                "Click2": config.CLICK_REGION_2,
                "BaitRegionCheck": config.BAIT_REGION, # Region used for bait check
                "Temperature": config.CELSIUS_REGION,
                "WeatherDesc": config.WEATHER_DESC_REGION,
                "MapName": config.MAP_REGION,
                "GameTime": config.GAME_TIME_REGION,
                "FishName": config.FISH_NAME_REGION,
                "FishWeight": config.WEIGHT_REGION,
                "BaitInfo": config.BAIT_REGION # Same region, different conceptual name
            }

            for name, coords in all_regions.items():
                color = REGION_COLORS.get(name, 'gray') # Default to gray if not defined
                if coords and len(coords) == 4: # Ensure coords are valid
                     manager.add_region_frame(name, coords, color=color, thickness=2)
                else:
                     print(f"Warning: Invalid or missing coordinates for region '{name}' in config.py. Skipping frame.")
        else:
            print("Skipping creation of region frame overlays.")


        # Start the worker thread
        worker_thread = threading.Thread(target=image_processor_worker, args=(manager,), daemon=True)
        worker_thread.start()

        # Start the main logging logic in a separate thread (no fishing_type needed)
        main_logger_thread = threading.Thread(target=main, args=(manager, stop_main_event), daemon=True)
        main_logger_thread.start()

        # Run the overlay manager's main loop in the primary thread
        # This will block until the overlay windows are closed (either text or frames)
        manager.run()

    except Exception as e:
        print(f"Error during setup or overlay execution: {e}")
        # Ensure stop events are set if setup fails badly
        stop_main_event.set()
        stop_worker_event.set()
    finally:
        # --- Graceful Shutdown Sequence ---
        # This block runs when manager.run() exits (e.g., KeyboardInterrupt in manager, window closed)
        print("Initiating shutdown from main execution block...")
        if manager:
            manager.update_queue.put("Shutting down...") # Try to update overlay

        # 1. Signal main logger thread to stop
        print("Signaling main logger thread to stop...")
        stop_main_event.set()
        if main_logger_thread and main_logger_thread.is_alive():
            main_logger_thread.join(timeout=2) # Wait briefly
            if main_logger_thread.is_alive():
                print("Main logger thread did not stop gracefully.")

        # 2. Signal worker thread to stop
        print("Signaling worker thread to stop...")
        stop_worker_event.set()
        processing_queue.put(None) # Unblock worker if waiting
        if worker_thread and worker_thread.is_alive():
            worker_thread.join(timeout=5) # Wait for worker
            if worker_thread.is_alive():
                print("Worker thread did not stop gracefully.")

        # 3. Unhook keyboard listener
        print("Unhooking keyboard listener...")
        keyboard.unhook_all() # Clean up hotkeys

        # 4. Close overlays (manager's run() finally block should also do this)
        if manager:
            manager.close_all()

        print("Shutdown complete.")
