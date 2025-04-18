import time
import os
import datetime
import uuid
import shutil
import sys
import re # Import regex module
import queue
import threading
# from dotenv import load_dotenv # Removed .env loading

# Third-party libraries
import pyautogui
import psutil

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

    # 2. Upscale
    print(f"Upscaling {region_name} image...")
    upscale_success, upscale_error_msg = image_utils.upscale_image(original_img_path, upscaled_img_path)
    if not upscale_success:
        error_details = f"Reason: {upscale_error_msg}" if upscale_error_msg else "Unknown reason."
        print(f"Failed to upscale {region_name}. {error_details} Attempting OCR on original.")
        manager.update_queue.put(f"{region_name}: Upscale Failed ({error_details[:50]}...), OCR on original...") # Truncate long errors for overlay
        # Fallback: try OCR on the original image if upscale fails
        ocr_texts, raw_ocr = screen_utils.perform_ocr(original_img_path)
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
    ocr_texts, raw_ocr = screen_utils.perform_ocr(upscaled_img_path)
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
            match = re.search(r'[a-zA-Z]', joined_text)
            processed_text = joined_text[match.start():].strip() if match else joined_text
            print(f"Cleaned WeatherDesc OCR: {processed_text}")
        elif region_name == "Temperature":
             if re.search(r'\d', joined_text) and not joined_text.upper().endswith('C'):
                cleaned_temp = re.sub(r'[^\d.]+$', '', joined_text).strip()
                processed_text = cleaned_temp + "°C" if cleaned_temp else joined_text
             else:
                processed_text = joined_text
             print(f"Formatted Temperature OCR: {processed_text}")
        else:
             processed_text = joined_text

        log_data[region_name] = processed_text
        print(f"OCR Result ({region_name}): {processed_text}") # Log processed text
        # Add successful OCR results to overlay text
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


# --- Main Execution ---

def main(manager, stop_event): # Accept manager and stop event
    """Main logging loop."""
    # Removed .env loading
    # load_dotenv()
    # print("Loaded environment variables from .env file (if present).")

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
    try:
        while not stop_event.is_set(): # Check stop event
            # Check if game process is running
            if not is_process_running(config.GAME_PROCESS_NAME):
                status_msg = f"Waiting for {config.GAME_PROCESS_NAME}..."
                print(status_msg + " (Check every 5s)")
                manager.update_queue.put(status_msg)
                time.sleep(5)
                continue # Skip the rest of the loop iteration

            # Check for "keep" in the trigger region
            manager.update_queue.put("Monitoring for 'keep'...")
            if screen_utils.check_text_in_region(config.TRIGGER_REGION, "keep"):
                print("\n>>> 'keep' detected! Starting action sequence. <<<")
                manager.update_queue.put("'keep' detected! Starting...")

                # --- Action Sequence ---
                input_utils.press_key('space')
                input_utils.delay(config.SHORT_DELAY) # Small delay after space

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

                    # --- Log comprehensive data ONLY if all processing succeeded ---
                    if not any_processing_failed:
                        print("All processing successful. Logging comprehensive data...")
                        manager.update_queue.put("Logging successful catch...")
                        log_utils.append_to_comprehensive_log(catch_data)
                    else:
                        print("One or more processing steps failed or timed out. Skipping comprehensive log entry.")
                        manager.update_queue.put("Processing failed. Skipping log.")
                    # --- Discord sending removed ---
                    # log_utils.send_to_discord(catch_data)

                    print("\n>>> Action sequence complete (processing done after Esc). Resuming monitoring. <<<")
                    # Add a delay AFTER completing the sequence to prevent immediate re-triggering
                    print("Adding post-sequence delay...")
                    manager.update_queue.put("Sequence complete. Delaying...")
                    # Use a loop for the delay to check stop_event periodically
                    delay_end_time = time.time() + 5.0
                    while time.time() < delay_end_time:
                        if stop_event.is_set(): break
                        time.sleep(0.1)


                else: # Keepnet not found
                    print("Timeout waiting for 'Keepnet'. Aborting sequence.")
                    manager.update_queue.put("Timeout waiting for 'Keepnet'.")
                    # Optional: Press Esc to back out if stuck
                    input_utils.press_key('esc')

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

            try:
                # 1. Upscale
                print(f"[Worker] Upscaling {region_name} image...")
                upscale_success, upscale_error_msg = image_utils.upscale_image(original_path, upscaled_path)

                if not upscale_success:
                    error_details = f"Reason: {upscale_error_msg}" if upscale_error_msg else "Unknown reason."
                    print(f"[Worker] Failed to upscale {region_name}. {error_details} Attempting OCR on original.")
                    overlay_lines[-1] = f"{region_name}: Upscale Failed, OCR Orig..." # Update last status
                    overlay_lines.append(f"  Error: {upscale_error_msg}") # Add specific error to overlay
                    manager.update_queue.put("\n".join(overlay_lines))
                    # Determine allowlist for fallback OCR
                    if region_name == "FishWeight":
                        allowlist_for_ocr = '0123456789g'
                    elif region_name == "GameTime":
                        allowlist_for_ocr = '0123456789.' # Added dot for GameTime
                    else:
                        allowlist_for_ocr = None
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
                    # 2. Perform OCR on Upscaled Image
                    print(f"[Worker] Performing OCR on upscaled {region_name} image...")
                    overlay_lines[-1] = f"{region_name}: Upscaled, OCR..." # Update last status
                    manager.update_queue.put("\n".join(overlay_lines))
                    # Determine allowlist for OCR
                    if region_name == "FishWeight":
                        allowlist_for_ocr = '0123456789g'
                    elif region_name == "GameTime":
                        allowlist_for_ocr = '0123456789.' # Added dot for GameTime
                    else:
                        allowlist_for_ocr = None
                    ocr_texts, raw_ocr = screen_utils.perform_ocr(upscaled_path, allowlist=allowlist_for_ocr)
                    if not ocr_texts:
                        print(f"[Worker] OCR failed for upscaled {region_name}.")
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
                        elif region_name == "GameTime":
                            # Format HHMM to HH:MM only if exactly 4 digits, otherwise use cleaned text
                            cleaned_time = re.sub(r'[^\d.]', '', joined_text) # Keep only digits and dot
                            if len(cleaned_time) == 4 and cleaned_time.isdigit():
                                processed_text = f"{cleaned_time[:2]}:{cleaned_time[2:]}"
                            else:
                                processed_text = cleaned_time # Use the cleaned result (e.g., "21.35")
                        elif region_name == "WeatherDesc":
                            match = re.search(r'[a-zA-Z]', joined_text)
                            processed_text = joined_text[match.start():].strip() if match else joined_text
                        elif region_name == "Temperature": # Temp is usually processed sync, but keep logic
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

                # Keep upscaled images as requested previously
                # if not upscale_success and os.path.exists(upscaled_path):
                #     try:
                #         os.remove(upscaled_path)
                #     except Exception as e:
                #         print(f"[Worker] Error deleting failed upscaled image {upscaled_path}: {e}")

                # Signal task completion *inside* the finally block
                processing_queue.task_done()

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

    try:
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

        # Start the main logging logic in a separate thread
        main_logger_thread = threading.Thread(target=main, args=(manager, stop_main_event,), daemon=True)
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

        # 3. Close overlays (manager's run() finally block should also do this)
        if manager:
            manager.close_all()

        print("Shutdown complete.")
