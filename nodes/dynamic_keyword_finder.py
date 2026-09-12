import re


class DynamicKeywordFinder:
    """Find multiple keywords in text and return one BOOLEAN per keyword.

    The frontend creates keyword_1 .. keyword_N dynamically.  The backend
    deliberately keeps the execution contract simple: keyword values arrive
    through **kwargs and only the active field_count entries are evaluated.
    """

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
            "optional": {
                **{
                    f"keyword_{i}": ("STRING", {"default": "", "multiline": False})
                    for i in range(1, cls.MAX_FIELDS + 1)
                }
            },
        }

    # The frontend keeps the visible output count equal to field_count.
    # The backend contract has a fixed maximum of BOOLEAN outputs, which keeps
    # output typing stable and avoids dynamic type mutation.
    RETURN_TYPES = ("BOOLEAN",) * MAX_FIELDS
    RETURN_NAMES = tuple(f"keyword_{i}_found" for i in range(1, MAX_FIELDS + 1))
    OUTPUT_IS_LIST = (False,) * MAX_FIELDS

    FUNCTION = "find"
    CATEGORY = "text/Dynamic"

    def find(self, text, field_count, whole_word=True, **kwargs):
        text = text or ""
        count = max(1, min(self.MAX_FIELDS, int(field_count)))
        results = []

        for i in range(1, self.MAX_FIELDS + 1):
            keyword = (kwargs.get(f"keyword_{i}", "") or "").strip()

            if i > count or not keyword:
                results.append(False)
                continue

            if whole_word:
                pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
            else:
                pattern = re.escape(keyword)

            results.append(re.search(pattern, text, flags=re.IGNORECASE) is not None)

        return tuple(results)

