import re


class DynamicOptionalInputs(dict):
    """Allow ComfyUI graph validation to accept dynamically named widgets."""

    def __contains__(self, key):
        return (
            isinstance(key, str)
            and (
                re.fullmatch(r"keyword_\d+", key)
                or re.fullmatch(r"expansion_\d+", key)
            )
        ) or super().__contains__(key)

    def __getitem__(self, key):
        if re.fullmatch(r"keyword_\d+", str(key)):
            return ("STRING", {"default": "", "multiline": False})
        if re.fullmatch(r"expansion_\d+", str(key)):
            return ("STRING", {"default": "", "multiline": True})
        return super().__getitem__(key)


class DynamicReturnTypes(tuple):
    """Expose only the two fixed outputs to the frontend, while allowing
    backend validation to resolve any dynamically-created output index.

    ComfyUI serializes the tuple's actual contents for /object_info, so the
    frontend sees only the two fixed outputs. Python-side output validation can
    still index beyond those two slots for dynamic outputs.
    """

    def __new__(cls):
        return super().__new__(cls, ("STRING", "STRING"))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return super().__getitem__(index)
        if index < 0:
            return super().__getitem__(index)
        if index < 2:
            return super().__getitem__(index)

        dynamic_index = index - 2
        return "BOOLEAN" if dynamic_index % 2 else "STRING"


class DynamicReturnNames(tuple):
    """Keep the serialized node definition to the two fixed outputs."""

    def __new__(cls):
        return super().__new__(cls, ("text", "matched_keywords"))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return super().__getitem__(index)
        if index < 0:
            return super().__getitem__(index)
        if index < 2:
            return super().__getitem__(index)
        dynamic_index = index - 2
        field = dynamic_index // 2 + 1
        suffix = "found" if dynamic_index % 2 else "text"
        return f"keyword_{field}_{suffix}"


class KeywordExpander:
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
                "replace_all": ("BOOLEAN", {"default": True}),
            },
            "optional": DynamicOptionalInputs(),
        }

    # IMPORTANT: only two outputs are advertised in object_info.  Dynamic
    # output indices are resolved by DynamicReturnTypes.__getitem__ on the
    # backend.  This prevents ComfyUI from calculating the node's minimum size
    # for all 20 fields on initial construction.
    RETURN_TYPES = DynamicReturnTypes()
    RETURN_NAMES = DynamicReturnNames()
    # Execution can return up to 40 dynamic scalar outputs after the two fixed
    # outputs.  This is consumed by merge_result_data; it does not enlarge the
    # serialized frontend output definition.
    OUTPUT_IS_LIST = (False,) * (2 + 20 * 2)

    FUNCTION = "expand"
    CATEGORY = "text/Dynamic"

    def expand(self, text, field_count, whole_word=True, replace_all=True, **kwargs):
        result = text
        matched = []
        outputs = []

        for i in range(1, self.MAX_FIELDS + 1):
            keyword = (kwargs.get(f"keyword_{i}", "") or "").strip()
            expansion = kwargs.get(f"expansion_{i}", "") or ""

            if i > field_count or not keyword:
                outputs.extend(["", False])
                continue

            pattern = (
                r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
                if whole_word else re.escape(keyword)
            )
            count = 0 if replace_all else 1

            new_result, n = re.subn(
                pattern,
                lambda _m: expansion,
                result,
                count=count,
                flags=re.IGNORECASE,
            )

            found = n > 0
            if found:
                result = new_result
                matched.append(keyword)

            outputs.extend([expansion if found else "", found])

        return tuple([result, ", ".join(matched)] + outputs)
