
class ListGetItem:
    """
    List Get Item
    --------------
    Pulls a single element out of a list-typed output (like
    DynamicKeywordExpander's `results` / `found` / `expansions`) by index,
    so it can be plugged into a node that expects one plain value instead
    of a list.

    Accepts and returns any type ("*"), so the same node works for STRING,
    BOOLEAN, or anything else. `index` is clamped into range, so an
    out-of-range index returns the last item rather than erroring.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("*", {}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
            }
        }

    # INPUT_IS_LIST=True so `list_input` arrives as the full Python list
    # rather than being fanned out one call per item.
    INPUT_IS_LIST = True

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("item",)

    FUNCTION = "get_item"
    CATEGORY = "text/keyword_expander"

    def get_item(self, list_input, index):
        # With INPUT_IS_LIST=True, every input arrives wrapped in a list --
        # including plain widget values like `index` -- so unwrap it first.
        items = list_input[0] if len(list_input) == 1 and isinstance(list_input[0], list) else list_input
        idx = index[0] if isinstance(index, list) else index

        if not items:
            return (None,)

        idx = max(0, min(len(items) - 1, int(idx)))
        return (items[idx],)
