import pyautogui
import time
import random

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

def press_key(key, presses=1, interval=0.1):
    """Presses the specified key."""
    try:
        for _ in range(presses):
            pyautogui.press(key)
            time.sleep(interval)
        print(f"Pressed key: {key} ({presses}x)")
    except Exception as e:
        print(f"Error pressing key {key}: {e}")

def click_at(x, y):
    """Clicks the left mouse button at the specified coordinates."""
    try:
        pyautogui.click(x, y)
        print(f"Clicked at: ({x}, {y})")
    except Exception as e:
        print(f"Error clicking at ({x}, {y}): {e}")

def click_within_region(region):
    """Clicks the left mouse button at a random point within the specified region."""
    left, top, width, height = region
    try:
        # Calculate a random point within the region
        x = left + random.randint(0, width - 1)
        y = top + random.randint(0, height - 1)
        click_at(x, y)
    except Exception as e:
        print(f"Error clicking within region {region}: {e}")

def delay(seconds):
    """Pauses execution for the specified number of seconds."""
    print(f"Delaying for {seconds} seconds...")
    time.sleep(seconds)
    print("Delay finished.")

if __name__ == '__main__':
    # Example usage (for testing)
    print("Testing input utils...")
    delay(2)
    # Example: Press 'a'
    # press_key('a')
    # delay(1)
    # Example: Click near screen center (adjust coords for your screen)
    # screen_width, screen_height = pyautogui.size()
    # click_at(screen_width // 2, screen_height // 2)
    # delay(1)
    # Example: Click within a dummy region
    # dummy_region = (100, 100, 50, 50)
    # click_within_region(dummy_region)
    print("Input utils test finished.")
