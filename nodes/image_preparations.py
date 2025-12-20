import torch
import numpy as np
from PIL import Image, ImageChops
import math

try:
    from server import PromptServer
except:
    PromptServer = None

ASPECTS = {
    "1:1 [ Perfect Square ]": (1, 1),
    "2:3 [ Classic Portrait ]": (2, 3),
    "3:4 [ Golden Ratio ]": (19, 26),
    "9:16 [ Slim Vertical ]": (17, 30),
    "9:21 [ Ultra Tall ]": (15, 34),
    "3:2 [ Golden Landscape ]": (3, 2),
    "16:9 [ Panorama ]": (30, 17),
    "21:9 [ Epic Ultrawide ]": (34, 15),
}


class Image_Preparations:
    display_name = "Image Preparations"
    PAD_COLOR = (255, 0, 207)  # Fixed pad color (RGB)

    @classmethod
    def INPUT_TYPES(s):
        megapixel_options = [f"{i / 10:.1f}" for i in range(10, 26)]
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixel": (megapixel_options, {"default": "2.0"}),
                "divisible_by": (["8", "16", "32", "64"], {"default": "64"}),
                "mode": (["resize", "pad"], {"default": "resize"}),
                "delta_percent": (
                "FLOAT", {"default": 5.0, "min": -30.0, "max": 30.0, "step": 0.5}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INFO_TEXT", "WIDTH", "HEIGHT")
    RETURN_NAMES = ("Processed Image", "Aspect Info", "Width", "Height")
    FUNCTION = "prepare"
    CATEGORY = "image/processing"

    def NODE_INFO(self):
        return getattr(self, "info", "Aspect: N/A")

    def prepare(self, image, megapixel, divisible_by, mode, delta_percent, unique_id):
        np_img = (image[0].cpu().numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(np_img)

        # Default Delta
        delta_percent = delta_percent - 6 if delta_percent < 0 else delta_percent + 6

        # Trim uniform border
        pil_img = self.trim_uniform_border(pil_img, tolerance=10)

        # Detect nearest Flux aspect
        w, h = pil_img.size
        ratio = w / h
        nearest_name = None
        nearest_diff = float("inf")
        nearest_w, nearest_h = 1, 1
        for name, (aw, ah) in ASPECTS.items():
            ar = aw / ah
            diff = abs(ratio - ar)
            if diff < nearest_diff:
                nearest_diff = diff
                nearest_name = name
                nearest_w, nearest_h = aw, ah
        detected_info = f"Nearest Flux Aspect : {nearest_name}"
        self.info = detected_info

        # Compute target dimensions
        mp_value = float(megapixel) * 1_044_480
        div = int(divisible_by)

        if delta_percent <= 0.0:
            # Calculate one dimension first (Height in this case)
            target_h = int(math.sqrt(mp_value * nearest_h / nearest_w))
            target_h = (target_h // div) * div  # Make divisible

            # Calculate height based on exact aspect ratio
            target_w = int(round(target_h * nearest_w / nearest_h))
            target_w = (target_w // div) * div  # Make divisible
            new_w = target_w
            new_h = max(1, int(round(target_h * (1 + (-delta_percent) / 100.0))))
        else:
            # Calculate one dimension first (width in this case)
            target_w = int(math.sqrt(mp_value * nearest_w / nearest_h))
            target_w = (target_w // div) * div  # Make divisible

            # Calculate height based on exact aspect ratio
            target_h = int(round(target_w * nearest_h / nearest_w))
            target_h = (target_h // div) * div  # Make divisible
            new_w = max(1, int(round(target_w * (1 + delta_percent / 100.0))))
            new_h = target_h

        # if new_w < nearest_w:
        #      new_w = nearest_w
        # if new_h < nearest_h:
        #      new_h = nearest_h

        resized_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Resize image
        # resized_img = pil_img.resize((target_w, target_h), Image.LANCZOS)

        # Apply delta_percent based on orientation
        # if delta_percent != 0.0:
        # w_res, h_res = resized_img.size

        # if h_res >= w_res:
        #     # Portrait → modify width
        #     new_w = max(1, int(round(w_res * (1 + delta_percent / 100.0))))
        #     new_h = h_res
        # else:
        #     # Landscape → modify height
        #     new_w = w_res
        #     new_h = max(1, int(round(h_res * (1 + delta_percent / 100.0))))

        # Pad if needed
        if mode == "pad":
            final_img = Image.new("RGB", (target_w, target_h), self.PAD_COLOR)
            offset_x = (target_w - resized_img.width) // 2
            offset_y = (target_h - resized_img.height) // 2
            final_img.paste(resized_img, (offset_x, offset_y))
        else:
            final_img = resized_img

        # Convert back to tensor
        np_resized = np.array(final_img).astype("float32") / 255.0
        out = torch.from_numpy(np_resized).unsqueeze(0)
        final_w, final_h = final_img.size

        # print(f"{unique_id}:[Image Preparations] {detected_info} | {megapixel} MP | mode={mode} | delta={delta_percent}% -> {final_w}x{final_h} (÷{div})")
        # Progress UI
        if unique_id and PromptServer is not None:
            try:
                num_elements = out.numel()
                element_size = out.element_size()
                memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                PromptServer.instance.send_progress_text(
                    f"<tr><td><b>{out.shape[2]}</b> x <b>{out.shape[1]}</b></td> | <td>Nearest Flux Aspect : <b>{nearest_name}</b></td></tr>",
                    unique_id
                )
            except:
                pass
        # Add debug prints
        print(f"Original size: {w}x{h}, ratio: {ratio}")
        print(f"Target aspect: {nearest_w}:{nearest_h}, ratio: {nearest_w / nearest_h}")
        print(f"Calculated dimensions: {target_w}x{target_h}, ratio: {target_w / target_h}")
        print(f"Final dimensions: {final_w}x{final_h}, ratio: {final_w / final_h}")

        return (out, detected_info, final_w, final_h)

    @staticmethod
    def trim_uniform_border(img, tolerance=8, patch_size=8):
        np_img = np.array(img)
        h, w = np_img.shape[:2]

        # extract corner patches (all lowercase)
        tl = np_img[0:patch_size, 0:patch_size]  # top-left
        tr = np_img[0:patch_size, w - patch_size:w]  # top-right
        bl = np_img[h - patch_size:h, 0:patch_size]  # bottom-left
        br = np_img[h - patch_size:h, w - patch_size:w]  # bottom-right

        def median_color(block):
            flat = block.reshape(-1, block.shape[-1])
            return tuple(np.median(flat, axis=0).astype(int))

        # compute median colors for the corners
        c_tl = median_color(tl)
        c_tr = median_color(tr)
        c_bl = median_color(bl)
        c_br = median_color(br)

        def close(c1, c2):
            return all(abs(a - b) <= tolerance for a, b in zip(c1, c2))

        # check if border is uniform
        if close(c_tl, c_tr) and close(c_tl, c_bl) and close(c_tl, c_br):
            border_color = c_tl

            bg = Image.new(img.mode, img.size, border_color)
            diff = ImageChops.difference(img, bg)
            bbox = diff.getbbox()

            if bbox:
                return img.crop(bbox)

        return img


NODE_CLASS_MAPPINGS = {
    "Image_Preparations": Image_Preparations
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image_Preparations": "Image Preparations"
}
