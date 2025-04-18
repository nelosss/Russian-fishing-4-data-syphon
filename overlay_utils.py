import tkinter as tk
import queue

# --- Text Overlay ---

class OCROverlay:
    """
    Manages a simple, transparent, always-on-top Tkinter window
    to display OCR text results.
    """
    def __init__(self, root, update_queue): # Accept root window
        self.root = root # Use the main root window
        self.update_queue = update_queue
        self.label_var = tk.StringVar()

        # Create a Toplevel window for the text overlay
        self.window = tk.Toplevel(self.root)
        self.window.overrideredirect(True)
        self.window.wm_attributes("-topmost", True)

        # Make window background transparent
        transparent_color = 'magenta' # Or another unlikely color
        self.window.wm_attributes("-transparentcolor", transparent_color)
        self.window.config(bg=transparent_color)

        # Label to display text
        self.label = tk.Label(
            self.window,
            textvariable=self.label_var,
            font=("Arial", 10, "bold"),
            bg=transparent_color,
            fg="yellow",
            justify=tk.LEFT,
            anchor='nw'
        )
        self.label.pack(padx=5, pady=5)

        # Initial position (adjust as needed)
        self.window.geometry("+10+10")
        self.label_var.set("OCR Overlay Initialized...")

        # Start checking the queue for updates
        self.check_queue()

    def check_queue(self):
        """Checks the update queue for new text and schedules the next check."""
        try:
            while True:
                message = self.update_queue.get_nowait()
                if isinstance(message, str): # Simple text update
                     self.label_var.set(message)
                # Add handling for other message types if needed later
        except queue.Empty:
            pass

        # Schedule the next check using the root window
        self.root.after(100, self.check_queue)

    # No run() method needed here, main loop is handled by the root window
    # def run(self):
    #     pass

    def close(self):
        """Closes the text overlay window."""
        print("Closing Text Overlay window...")
        try:
            self.window.destroy()
        except tk.TclError:
            print("Text Overlay window already closed or could not be destroyed.")
        except Exception as e:
            print(f"Error closing text overlay window: {e}")


# --- Region Frame Overlay ---

class RegionFrameOverlay:
    """
    Creates four thin Toplevel windows to form a colored frame
    around a specified screen region.
    """
    def __init__(self, root, region_coords, color='red', thickness=2):
        self.root = root # Use the main hidden root window
        self.frames = []
        self.region = region_coords
        self.color = color
        self.thickness = thickness
        self._create_frames()

    def _create_window(self, geometry):
        """Helper to create a single frame window."""
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.wm_attributes("-topmost", True)
        # Make the window non-interactive (Windows specific)
        # This allows clicks to pass through the thin frame
        try:
            win.wm_attributes("-disabled", True)
            win.wm_attributes("-transparentcolor", "white") # Make white parts transparent
            win.config(bg=self.color) # Set the frame color
        except tk.TclError:
             print("Note: -disabled attribute might not work on non-Windows OS.")
             win.config(bg=self.color) # Still set color

        win.geometry(geometry)
        return win

    def _create_frames(self):
        """Creates the four Toplevel windows for the frame."""
        left, top, width, height = self.region
        t = self.thickness

        # Top frame
        geo_top = f"{width}x{t}+{left}+{top}"
        self.frames.append(self._create_window(geo_top))

        # Bottom frame
        geo_bottom = f"{width}x{t}+{left}+{top + height - t}"
        self.frames.append(self._create_window(geo_bottom))

        # Left frame
        geo_left = f"{t}x{height - 2*t}+{left}+{top + t}" # Adjust height to avoid overlap
        self.frames.append(self._create_window(geo_left))

        # Right frame
        geo_right = f"{t}x{height - 2*t}+{left + width - t}+{top + t}" # Adjust height
        self.frames.append(self._create_window(geo_right))

    def show(self):
        """Makes the frame windows visible."""
        for frame in self.frames:
            try:
                frame.deiconify() # Make visible if withdrawn
            except tk.TclError: pass # Ignore if already visible/destroyed

    def hide(self):
        """Hides the frame windows."""
        for frame in self.frames:
            try:
                frame.withdraw() # Hide the window
            except tk.TclError: pass # Ignore if already hidden/destroyed

    def close(self):
        """Destroys all frame windows."""
        print(f"Closing frame overlay for region {self.region}...")
        for frame in self.frames:
            try:
                frame.destroy()
            except tk.TclError:
                pass # Window might already be destroyed
            except Exception as e:
                print(f"Error closing frame window: {e}")
        self.frames = []


# --- Main Application Runner ---
# Helper to manage the root Tk instance and run the main loop

class OverlayManager:
    def __init__(self):
        self.root = tk.Tk()
        # Hide the root window completely
        self.root.withdraw()
        self.root.wm_attributes("-topmost", True) # Keep root potentially on top if needed

        self.text_overlay = None
        self.region_overlays = {} # Store region frames {name: RegionFrameOverlay}
        self.update_queue = queue.Queue()

    def create_text_overlay(self):
        self.text_overlay = OCROverlay(self.root, self.update_queue)

    def add_region_frame(self, name, region_coords, color='red', thickness=2):
        if name in self.region_overlays:
            self.region_overlays[name].close() # Close existing if any
        frame = RegionFrameOverlay(self.root, region_coords, color, thickness)
        self.region_overlays[name] = frame
        print(f"Created region frame '{name}' with color {color}")

    def run(self):
        if not self.text_overlay:
            print("Warning: Text overlay not created. Call create_text_overlay() first.")
            # Create a default one if needed? Or just run without text.
            self.create_text_overlay() # Create a default one

        print("Starting Overlay Manager (Tkinter main loop)...")
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received in OverlayManager.")
        finally:
            self.close_all()

    def close_all(self):
        print("Closing all overlays...")
        if self.text_overlay:
            self.text_overlay.close()
        for name, frame in list(self.region_overlays.items()): # Iterate over copy
            frame.close()
        self.region_overlays = {}
        try:
            self.root.destroy()
            print("Root Tk window destroyed.")
        except tk.TclError:
            print("Root Tk window already destroyed or could not be destroyed.")
        except Exception as e:
            print(f"Error destroying root Tk window: {e}")

# Example usage (if run directly)
if __name__ == '__main__':
    manager = OverlayManager()

    # Create text overlay
    manager.create_text_overlay()

    # Define some test regions and colors
    regions = {
        "TopLeft": ((10, 50, 200, 100), 'red'),
        "Center": ((300, 300, 150, 50), 'blue'),
        "BottomRight": ((600, 500, 100, 150), 'green')
    }

    # Add region frames
    for name, (coords, color) in regions.items():
        manager.add_region_frame(name, coords, color)

    # Simulate text updates
    def simulate_updates():
        import time
        count = 0
        texts = ["Status: Running", "Detected: Item A (0.9)", "Detected: Item B (0.7)", "Status: Idle"]
        while True:
            try:
                text = texts[count % len(texts)]
                print(f"Simulating text update: {text}")
                manager.update_queue.put(text)
                count += 1
                time.sleep(3)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Simulation error: {e}")
                break
        print("Simulation thread stopping.")
        # manager.close_all() # Shutdown is handled by finally block in run()

    import threading
    update_thread = threading.Thread(target=simulate_updates, daemon=True)
    update_thread.start()

    manager.run() # Start the overlay manager's main loop
