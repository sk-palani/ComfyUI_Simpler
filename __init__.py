# This file can be empty or include imports for ComfyUI to recognize the module

from .nodes.list_get_item import ListGetItem
from .nodes.dynamic_keyword_expander import DynamicKeywordExpander
from .nodes.input_selector import InputSelector
from .nodes.image_preparations import Image_Preparations
from .nodes.input_image import Input_Image
from .nodes.dynamic_metadata_extractor import DynamicMetadataExtractor
from .nodes.dynamic_keyword_finder import DynamicKeywordFinder

# Combine mappings from all node modules
NODE_CLASS_MAPPINGS = {
    "ListGetItem": ListGetItem,
    "InputSelector": InputSelector,
    "Image_Preparations": Image_Preparations,
    "Input_Image": Input_Image,
    "DynamicKeywordExpander": DynamicKeywordExpander,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ListGetItem": "List Get Item",
    "InputSelector": "Input Selector",
    "Image_Preparations": "Image Preparations",
    "Input_Image": "Input Image",
    "DynamicKeywordExpander": "Dynamic Keyword Expander",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
