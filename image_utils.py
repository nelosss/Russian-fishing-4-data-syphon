import subprocess
import os
import config  # Import configuration variables
import time

# Default model set to "models-cunet" based on direct command line test
def upscale_image(input_path, output_path, scale_factor=2, model="models-cunet", gpu_id=0):
    """
    Upscales an image using the waifu2x-ncnn-vulkan executable.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the upscaled output image file.
        scale_factor (int): Upscaling factor (e.g., 2 for 2x).
        model (str): The model directory name within the waifu2x folder (e.g., "models-cunet").
        gpu_id (int): The GPU ID to use for processing.

    Returns:
        tuple[bool, str | None]: A tuple containing:
            - bool: True if upscaling was successful, False otherwise.
            - str | None: An error message if unsuccessful, otherwise None.
    """
    error_prefix = "[Upscale Error] "
    if not os.path.exists(config.WAIFU2X_EXECUTABLE):
        msg = f"Waifu2x executable not found at '{config.WAIFU2X_EXECUTABLE}'"
        print(error_prefix + msg)
        return False, msg

    if not os.path.exists(input_path):
        msg = f"Input image not found at '{input_path}'"
        print(error_prefix + msg)
        return False, msg

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir: # Check if output_dir is not empty (i.e., not the current dir)
        os.makedirs(output_dir, exist_ok=True)

    # Convert input/output paths to absolute paths
    abs_input_path = os.path.abspath(input_path)
    abs_output_path = os.path.abspath(output_path)

    command = [
        config.WAIFU2X_EXECUTABLE,
        "-i", abs_input_path,
        "-o", abs_output_path,
        "-s", str(scale_factor),
        "-m", model, # Using model="cunet" by default now
        "-g", str(gpu_id),
        "-f", "png" # Specify output format as png
    ]

    print(f"Running waifu2x: {' '.join(command)}")
    try:
        # Determine the directory of the waifu2x executable
        executable_path = os.path.abspath(config.WAIFU2X_EXECUTABLE)
        executable_dir = os.path.dirname(executable_path)
        print(f"Setting waifu2x working directory to: {executable_dir}")

        # Use shell=True on Windows if direct execution fails, but be cautious
        # For now, try without shell=True as it's generally safer
        # Explicitly set the current working directory (cwd) for the subprocess
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW, # Hide console window
            cwd=executable_dir # Set the working directory
        )

        if result.returncode == 0:
            print(f"Successfully upscaled '{abs_input_path}' to '{abs_output_path}'")
            # Add a small delay to ensure file is fully written
            time.sleep(0.2)
            # Check existence using the absolute path
            if os.path.exists(abs_output_path) and os.path.getsize(abs_output_path) > 0:
                 print(f"Output file '{abs_output_path}' confirmed.")
                 return True, None # Success
            else:
                 error_msg = f"Output file '{abs_output_path}' not found or empty after upscaling.\nStdout: {result.stdout}\nStderr: {result.stderr}"
                 print(error_prefix + error_msg)
                 return False, error_msg
        else:
            error_msg = f"waifu2x execution failed (Return Code: {result.returncode}).\nCommand: {' '.join(command)}\nStdout: {result.stdout}\nStderr: {result.stderr}"
            print(error_prefix + error_msg)
            return False, error_msg
    except FileNotFoundError:
        error_msg = f"Command not found. Is '{config.WAIFU2X_EXECUTABLE}' in PATH or script directory?"
        print(error_prefix + error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error running waifu2x: {e}"
        print(error_prefix + error_msg)
        return False, error_msg

if __name__ == '__main__':
    # Example Usage (Requires waifu2x executable and a test image)
    print("Testing image utils...")
    # Create dummy input file for testing structure (replace with actual image for real test)
    TEST_INPUT = "test_input.png"
    TEST_OUTPUT = os.path.join(config.TEMP_IMAGE_DIR, "test_output_upscaled.png")

    # Create a dummy temp dir if it doesn't exist
    if not os.path.exists(config.TEMP_IMAGE_DIR):
        os.makedirs(config.TEMP_IMAGE_DIR)

    # Create a placeholder input file if it doesn't exist
    if not os.path.exists(TEST_INPUT):
        try:
            with open(TEST_INPUT, "w") as f:
                f.write("dummy")
            print(f"Created dummy input file: {TEST_INPUT}")
        except IOError as e:
            print(f"Could not create dummy input file: {e}")


    # Check if waifu2x exists before trying to upscale
    if os.path.exists(config.WAIFU2X_EXECUTABLE) and os.path.exists(TEST_INPUT):
         print(f"Attempting to upscale '{TEST_INPUT}' to '{TEST_OUTPUT}'...")
         # Note: This will likely fail if TEST_INPUT is not a valid image format
         # but it tests the subprocess call structure.
         success, error_msg = upscale_image(TEST_INPUT, TEST_OUTPUT)
         if success:
             print("Upscaling test reported success.")
             # Clean up dummy output
             # if os.path.exists(TEST_OUTPUT):
             #     os.remove(TEST_OUTPUT)
         else:
             print(f"Upscaling test failed: {error_msg}")
    else:
        if not os.path.exists(config.WAIFU2X_EXECUTABLE):
            print(f"Skipping upscale test: Waifu2x executable not found at '{config.WAIFU2X_EXECUTABLE}'")
        if not os.path.exists(TEST_INPUT):
             print(f"Skipping upscale test: Test input file '{TEST_INPUT}' not found.")


    # Clean up dummy input file
    # if os.path.exists(TEST_INPUT) and open(TEST_INPUT).read() == "dummy":
    #      os.remove(TEST_INPUT)

    print("Image utils test finished.")
