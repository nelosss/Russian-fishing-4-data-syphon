import os
from PIL import Image, ImageOps

# List of template image filenames to invert
# Ensure these files are in the same directory as this script
template_files = [
    "sunny.png",
    "cloudy.png",
    "rainy.png",
    "thunder.png",
    "sun_with_clouds.png",
    # Add any other weather symbol images you use here
]

print("Starting template image inversion...")

for filename in template_files:
    if not os.path.exists(filename):
        print(f"Warning: File not found, skipping: {filename}")
        continue

    try:
        print(f"Processing {filename}...")
        # Open the image
        img = Image.open(filename)

        # Ensure image has an alpha channel for proper inversion handling if needed
        # Convert to RGBA if it's not already, common for PNGs
        if img.mode != 'RGBA':
             img = img.convert('RGBA')

        # Separate RGB channels from Alpha channel
        rgb = img.split()[:3] # Get R, G, B channels
        alpha = img.split()[3] # Get Alpha channel

        # Invert the RGB channels only
        inverted_rgb = [ImageOps.invert(channel) for channel in rgb]

        # Merge the inverted RGB channels back with the original Alpha channel
        inverted_img = Image.merge('RGBA', inverted_rgb + [alpha])

        # Save the inverted image, overwriting the original
        inverted_img.save(filename)
        print(f"Successfully inverted and saved: {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Image inversion complete.")
