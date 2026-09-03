import os
import random
from PIL import Image
import torch
import numpy as np
from datetime import datetime


class Input_Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_1": ("STRING", {"default": ""}),
                "ordering_mode": (["random", "sequential_by_name", "sequential_by_modified_date"],
                                  {"default": "random"}),
            },
            "optional": {
                "directory_2": ("STRING", {"default": ""}),
                "directory_3": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999999}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "filename", "filepath")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, directory_1, ordering_mode, directory_2="", directory_3="", index=0):
        # Check directories in order
        directories = [directory_1, directory_2, directory_3]
        valid_directories = [d for d in directories if d and os.path.isdir(d)]

        if not valid_directories:
            raise ValueError("No valid directories provided")

        # Get all image files from valid directories
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff']
        image_files = []

        for directory in valid_directories:
            if not os.path.exists(directory):
                continue

            files_in_dir = []
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath) and os.path.splitext(filename)[1].lower() in image_extensions:
                    files_in_dir.append(filepath)

            # If we found files in this directory, we don't need to check the next ones
            if files_in_dir:
                image_files = files_in_dir
                break

        if not image_files:
            raise ValueError("No image files found in any of the provided directories")

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

        # Return image tensor, filename, and filepath
        filename = os.path.basename(selected_image)
        return (img_tensor, filename, selected_image)


# Add node to NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Input_Image": Input_Image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Input_Image": "Input Image"
}