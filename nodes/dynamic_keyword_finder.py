import re


class DynamicOptionalInputs(dict):
    """Accept keyword_1..keyword_20 as dynamically-created STRING inputs."""

    def __contains__(self, key):
        if isinstance(key, str) and re.fullmatch(r"keyword_\d+", key):
            return True
        return super().__contains__(key)

    def __getitem__(self, key):
        if isinstance(key, str) and re.fullmatch(r"keyword_\d+", key):
            return ("STRING", {"default": "", "multiline": False})
        return super().__getitem__(key)


class DynamicReturnTypes(tuple):
    """Keep object_info small while backend validation can resolve dynamic outputs."""

    def __new__(cls):
        return super().__new__(cls, ("BOOLEAN",))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return super().__getitem__(index)
        if index < 0:
            return super().__getitem__(index)
        return "BOOLEAN"


class DynamicReturnNames(tuple):
    def __new__(cls):
        return super().__new__(cls, ("keyword_1_found",))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return super().__getitem__(index)
        if index < 0:
            return super().__getitem__(index)
        return f"keyword_{index + 1}_found"


class DynamicKeywordFinder:
    MAX_FIELDS = 20

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "field_count": (
                    "INT",
                    {"default": 3, "min": 1, "max": cls.MAX_FIELDS, "step": 1},
                ),
                "whole_word": ("BOOLEAN", {"default": True}),
            },
            "optional": DynamicOptionalInputs(),
        }

    RETURN_TYPES = DynamicReturnTypes()
    RETURN_NAMES = DynamicReturnNames()
    OUTPUT_IS_LIST = (False,) * MAX_FIELDS
    FUNCTION = "find"
    CATEGORY = "text/Dynamic"

    def find(self, text, field_count, whole_word=True, **kwargs):
        text = text or ""
        count = max(1, min(self.MAX_FIELDS, int(field_count)))
        results = []

        for i in range(1, self.MAX_FIELDS + 1):
            if i > count:
                break

            keyword = (kwargs.get(f"keyword_{i}", "") or "").strip()
            if not keyword:
                results.append(False)
                continue

            if whole_word:
                pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
            else:
                pattern = re.escape(keyword)

            results.append(bool(re.search(pattern, text, flags=re.IGNORECASE)))

        return tuple(results)
