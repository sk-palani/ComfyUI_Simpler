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
    """Advertise the complete fixed output surface so ComfyUI knows the
    real type of every dynamic socket. The frontend may display only the
    currently active subset, but BOOLEAN Found sockets remain genuinely
    BOOLEAN to connection validation.
    """

    def __new__(cls):
        values = ["STRING", "STRING"]
        for _ in range(20):
            values.extend(("STRING", "BOOLEAN"))
        return super().__new__(cls, values)


class DynamicReturnNames(tuple):
    def __new__(cls):
        values = ["text", "matched_keywords"]
        for i in range(1, 21):
            values.extend((f"keyword_{i}_text", f"keyword_{i}_found"))
        return super().__new__(cls, values)


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
                "boolean_only": ("BOOLEAN", {"default": False}),
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

    def expand(self, text, field_count, whole_word=True, replace_all=True, boolean_only=False, **kwargs):
        result = text
        matched = []
        outputs = []

        for i in range(1, self.MAX_FIELDS + 1):
            keyword = (kwargs.get(f"keyword_{i}", "") or "").strip()
            expansion = kwargs.get(f"expansion_{i}", "") or ""

            if i > field_count or not keyword:
                outputs.extend((["", False] if not boolean_only else [False]))
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

            outputs.extend(([expansion if found else "", found] if not boolean_only else [found]))

        return tuple([result, ", ".join(matched)] + outputs)
