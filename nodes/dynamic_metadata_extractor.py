import re


class DynamicMetadataExtractor:
    """Extract @@KEY=value metadata with a user-selectable number of fields."""

    CATEGORY = "text/Dynamic"
    FUNCTION = "extract"
    MAX_FIELDS = 20

    # ComfyUI requires a statically declared return signature on the Python side.
    # The frontend trims the visible output slots to 2 * field_count.
    RETURN_TYPES = tuple(
        value_type
        for _ in range(MAX_FIELDS)
        for value_type in ("STRING", "BOOLEAN")
    )

    RETURN_NAMES = tuple(
        name
        for i in range(1, MAX_FIELDS + 1)
        for name in (f"field_{i}", f"field_{i}_found")
    )

    @classmethod
    def INPUT_TYPES(cls):
        # field_count is a real required input so ComfyUI graph validation can
        # always resolve it. The individual field widgets are optional inputs
        # with a fixed maximum; the JS extension only displays the selected ones.
        optional = {
            f"field_{i}": ("STRING", {"default": ""})
            for i in range(1, cls.MAX_FIELDS + 1)
        }
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text containing metadata such as @@AGE=23-27",
                    },
                ),
                "field_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": cls.MAX_FIELDS,
                        "step": 1,
                    },
                ),
            },
            "optional": optional,
        }

    @staticmethod
    def _extract(text, key):
        key = str(key or "").strip()
        if not key:
            return "", False

        # Literal key; value is everything after the first '=' on that line.
        pattern = rf"(?m)^[ \t]*@@{re.escape(key)}[ \t]*=[ \t]*(.*?)[ \t]*$"
        match = re.search(pattern, text or "")
        if match is None:
            return "", False

        return match.group(1).strip(), True

    def extract(self, text, field_count=3, **kwargs):
        results = []
        count = max(1, min(self.MAX_FIELDS, int(field_count or 1)))

        for i in range(1, self.MAX_FIELDS + 1):
            # Only the selected fields are meaningful. Return empty pairs for
            # the remaining statically-declared backend outputs.
            if i > count:
                results.extend(("", False))
                continue

            key = kwargs.get(f"field_{i}", "")
            value, found = self._extract(text, key)
            results.extend((value, found))

        return tuple(results)


NODE_CLASS_MAPPINGS = {
    "DynamicMetadataExtractor": DynamicMetadataExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicMetadataExtractor": "Dynamic Metadata Extractor",
}
