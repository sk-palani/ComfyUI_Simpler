import os
import random
from PIL import Image
import torch
import numpy as np
from datetime import datetime
import uuid


class Input_Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_1": ("STRING", {"default": "input/Next/Best"}),
            },
            "optional": {
                "directory_2": ("STRING", {"default": "input/Next"}),
                "directory_3": ("STRING", {"default": "input/Downloaded"}),
                "ordering_mode": (["random", "sequential_by_name", "sequential_by_modified_date"],
                                  {"default": "sequential_by_modified_date"}),
                "include_subdirectories": (["yes", "no"], {"default": "yes"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999999}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "filename", "source", "filepath", "subfolder")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, directory_1, directory_2="", directory_3="", ordering_mode="random",
                   include_subdirectories="no", index=0, seed=0):
        # Set up random seed
        # Use seed of 0 as an indicator to use current time for true randomness
        if seed == 0:
            current_time_seed = int(datetime.now().timestamp()) % (2 ** 32)
            random.seed(current_time_seed)
            np.random.seed(current_time_seed)
        else:
            # Otherwise use provided seed for reproducible randomness
            random.seed(seed)
            np.random.seed(seed)

        # Check directories in order
        directories = [directory_1, directory_2, directory_3]
        valid_directories = [d for d in directories if d and os.path.isdir(d)]

        # Get all image files from valid directories
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff']
        image_files = []
        active_directory = None

        for directory in valid_directories:
            if not os.path.exists(directory):
                continue

            files_in_dir = []

            # Decide whether to recursively search subdirectories
            if include_subdirectories == "yes":
                for root, _, files in os.walk(directory):
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        if os.path.isfile(filepath) and os.path.splitext(filename)[
                            1].lower() in image_extensions:
                            files_in_dir.append(filepath)
            else:
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath) and os.path.splitext(filename)[
                        1].lower() in image_extensions:
                        files_in_dir.append(filepath)

            # If we found files in this directory, we don't need to check the next ones
            if files_in_dir:
                image_files = files_in_dir
                active_directory = directory
                break

        if not image_files:
            # Instead of raising an error, generate a 1024x1024 darker random noise image
            print(
                "No image files found in any of the provided directories. Generating random noise image instead.")

            # Generate darker random noise (values between 0.0 and 0.3 for a darker image)
            noise_np = np.random.random((1024, 1024, 3)).astype(np.float32) * 0.3

            # Create a unique filename for the noise image
            unique_id = uuid.uuid4().hex[:8]
            filename = f"random_noise_{unique_id}.png"

            # Find a directory to save the noise image
            save_directory = None
            # Check directories in reverse order: first try directory_3, then directory_2, then directory_1
            for directory in [directory_3, directory_2, directory_1]:
                if directory and os.path.exists(directory):
                    save_directory = directory
                    break

            if save_directory:
                # Save the noise as an image file
                filepath = os.path.join(save_directory, filename)

                # Convert the float32 noise to uint8 image for saving
                noise_img = (noise_np * 255).astype(np.uint8)
                pil_img = Image.fromarray(noise_img)
                pil_img.save(filepath)

                print(f"Random noise image saved to: {filepath}")

                # For noise images, subfolder is empty as it's directly in the input directory
                subfolder = ""
                source = save_directory
            else:
                # If no valid directories exist, just return the noise without saving
                filepath = "random_noise_image"
                subfolder = ""
                source = ""
                print("No valid directories to save random noise image.")

            # Return the noise image tensor
            noise_tensor = torch.from_numpy(noise_np)[None,]
            return noise_tensor, filename, filepath, subfolder, source

        # Sort files based on ordering mode
        if ordering_mode == "random":
            random.shuffle(image_files)
            selected_image = image_files[index % len(image_files)]
        elif ordering_mode == "sequential_by_name":
            image_files.sort()
            selected_image = image_files[index % len(image_files)]
        elif ordering_mode == "sequential_by_modified_date":
            image_files.sort(key=lambda x: os.path.getmtime(x))
            selected_image = image_files[index % len(image_files)]
        else:
            raise ValueError(f"Unknown ordering mode: {ordering_mode}")

        # Load the image
        img = Image.open(selected_image)
        img = img.convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]

        # Return image tensor, filename, filepath and subfolder
        filename = os.path.basename(selected_image)

        # Extract subfolder as the path relative to the input directory
        if active_directory:
            # Add debugging prints
            print(f"DEBUG: selected_image = {selected_image}")
            print(f"DEBUG: active_directory = {active_directory}")

            # Normalize paths to handle different path separators
            norm_selected = os.path.normpath(selected_image)
            norm_active = os.path.normpath(active_directory)

            print(f"DEBUG: norm_selected = {norm_selected}")
            print(f"DEBUG: norm_active = {norm_active}")

            # Check if the selected image is within the active directory
            if norm_selected.startswith(norm_active):
                # Get the relative path from active directory to the selected image
                image_dir = os.path.dirname(norm_selected)
                rel_path = os.path.relpath(image_dir, norm_active)

                print(f"DEBUG: image_dir = {image_dir}")
                print(f"DEBUG: rel_path = {rel_path}")

                # If the relative path is ".", it means the file is directly in the input directory
                if rel_path == ".":
                    subfolder = ""
                else:
                    subfolder = rel_path

                print(f"DEBUG: subfolder = {subfolder}")
            else:
                # This should not happen unless there's an issue with path handling
                subfolder = ""
                print(f"DEBUG: norm_selected does not start with norm_active")

            # Set source as the active directory
            source = active_directory
        else:
            subfolder = ""
            source = ""
            print("DEBUG: No active directory")

        return img_tensor, filename, source, selected_image, subfolder


# Add node to NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Input_Image": Input_Image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Input_Image": "Input Image"
}
