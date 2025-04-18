import mss
import mss.tools
import numpy as np
import cv2
import easyocr
import os
import time
import config # Import configuration variables

# --- Initialize EasyOCR Reader ---
print("Initializing EasyOCR Reader...")
reader = None # Initialize reader to None first
try:
    # Force CPU initialization directly
    print("Attempting EasyOCR CPU initialization...")
    reader = easyocr.Reader(['en'], gpu=False) # Explicitly use CPU
    print("EasyOCR initialized successfully (CPU).")
except Exception as e_cpu:
    print(f"FATAL: EasyOCR CPU initialization failed: {e_cpu}")
    # Optionally, you could try GPU here as a fallback, but if CPU fails, GPU likely will too.
    # print("Attempting GPU initialization as fallback...")
    # try:
    #     reader = easyocr.Reader(['en'], gpu=True)
    #     print("EasyOCR initialized successfully (GPU fallback).")
    # except Exception as e_gpu:
    #     print(f"FATAL: EasyOCR GPU fallback initialization also failed: {e_gpu}")

# --- Screen Capture Functions ---

def capture_region(region, filename=None):
    """
    Captures a screenshot of the specified region.

    Args:
        region (tuple): A tuple (left, top, width, height) defining the screen area.
        filename (str, optional): If provided, saves the screenshot to this file path.

    Returns:
        numpy.ndarray: The captured image as a NumPy array (in BGR format for OpenCV compatibility),
                       or None if capture fails.
    """
    left, top, width, height = region
    monitor = {"top": top, "left": left, "width": width, "height": height}

    try:
        with mss.mss() as sct:
            sct_img = sct.grab(monitor)
            # Convert to NumPy array
            img = np.array(sct_img)
            # Convert RGB to BGR for OpenCV compatibility
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if filename:
                # Ensure the directory exists
                output_dir = os.path.dirname(filename)
                if output_dir:
                     os.makedirs(output_dir, exist_ok=True)
                # Save the image using OpenCV
                cv2.imwrite(filename, img_bgr)
                print(f"Screenshot saved to: {filename}")
            return img_bgr
    except mss.ScreenShotError as ex:
        print(f"Error capturing screen region {region}: {ex}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during screen capture: {e}")
        return None

# --- OCR Functions ---

def perform_ocr(image_path_or_array, allowlist=None):
    """
    Performs OCR on the given image file or NumPy array.

    Args:
        image_path_or_array (str or numpy.ndarray): Path to the image file or the image as a NumPy array.
        allowlist (str, optional): A string of characters to prioritize during OCR (e.g., '0123456789.'). Defaults to None.

    Returns:
        tuple: A tuple containing:
            - list: A list of detected text strings.
            - list: The raw OCR result from easyocr (list of tuples: [bbox, text, confidence]).
            Returns ([], []) if OCR fails or reader is not initialized.
    """
    if reader is None:
        print("Error: EasyOCR reader not initialized.")
        return [], []

    try:
        # Read the image (either from path or use the array directly)
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                print(f"Error: Image file not found for OCR: {image_path_or_array}")
                return [], []
            # Read image using OpenCV to handle potential path issues
            img = cv2.imread(image_path_or_array)
            if img is None:
                print(f"Error: Could not read image file: {image_path_or_array}")
                return [], []
        elif isinstance(image_path_or_array, np.ndarray):
            img = image_path_or_array
        else:
            print("Error: Invalid input type for OCR. Must be file path or NumPy array.")
            return [], []

        # Perform OCR
        # detail=1 provides bounding boxes and confidence scores
        ocr_kwargs = {'detail': 1, 'paragraph': False}
        if allowlist:
            ocr_kwargs['allowlist'] = allowlist
            print(f"Performing OCR with allowlist: '{allowlist}'")
        else:
            print("Performing OCR without specific allowlist.")

        result = reader.readtext(img, **ocr_kwargs)

        detected_texts = [text for (_, text, _) in result]
        print(f"OCR Result for '{'array input' if isinstance(image_path_or_array, np.ndarray) else os.path.basename(str(image_path_or_array))}': {detected_texts}") # Use basename for paths
        return detected_texts, result

    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return [], []

def check_text_in_region(region, target_text, confidence_threshold=0.5):
    """
    Captures a region, performs OCR, and checks if the target text is present.
    Case-insensitive comparison.

    Args:
        region (tuple): The screen region (left, top, width, height) to check.
        target_text (str): The text to search for (case-insensitive).
        confidence_threshold (float): Minimum confidence score for OCR results to be considered.

    Returns:
        bool: True if the target text is found with sufficient confidence, False otherwise.
    """
    print(f"Checking for text '{target_text}' in region {region}...")
    img_array = capture_region(region)
    if img_array is None:
        print("Failed to capture region for text check.")
        return False

    # Pass allowlist=None here as we usually want general detection for trigger text
    detected_texts, raw_results = perform_ocr(img_array, allowlist=None)

    if not raw_results:
        print("No text detected in region.")
        return False

    target_text_lower = target_text.lower()
    found = False
    for (bbox, text, confidence) in raw_results:
        text_lower = text.lower()
        print(f"  Detected: '{text}' (Confidence: {confidence:.2f})")
        # Simple substring check (case-insensitive)
        if target_text_lower in text_lower and confidence >= confidence_threshold:
            print(f"  Match found: '{text}' contains '{target_text}' with confidence {confidence:.2f}")
            found = True
            break # Stop after first match

    if not found:
        print(f"Target text '{target_text}' not found in region {region}.")

    return found


if __name__ == '__main__':
    # Example Usage (for testing - adjust region/text as needed)
    print("Testing screen utils...")

    if reader is None:
        print("Cannot run tests: EasyOCR Reader failed to initialize.")
    else:
        # Test capture (saves to temp dir)
        test_region = (0, 0, 200, 100) # Top-left corner
        temp_dir = config.TEMP_IMAGE_DIR
        os.makedirs(temp_dir, exist_ok=True)
        test_capture_file = os.path.join(temp_dir, "test_capture.png")
        print(f"\nAttempting to capture region {test_region} to {test_capture_file}")
        img_arr = capture_region(test_region, filename=test_capture_file)
        if img_arr is not None:
            print("Capture successful.")
            # Test OCR on the captured file
            print(f"\nAttempting OCR on {test_capture_file}")
            texts, raw = perform_ocr(test_capture_file)
            print("OCR Texts:", texts)
            # print("OCR Raw:", raw)
        else:
            print("Capture failed.")

        # Test text checking (replace 'your_target_text' with something likely on screen)
        # Be mindful this requires visual confirmation during testing
        check_region = (0, 0, 500, 100) # Wider top-left region
        text_to_find = "file" # Example: look for 'File' menu in an editor
        print(f"\nAttempting to find text '{text_to_find}' in region {check_region}")
        # Add a delay to allow switching to a window where the text might be visible
        print("Switch to a relevant window if needed...")
        time.sleep(3)
        found = check_text_in_region(check_region, text_to_find)
        print(f"Text '{text_to_find}' found: {found}")

    print("\nScreen utils test finished.")
