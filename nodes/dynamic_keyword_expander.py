import re

# ---------------------------------------------------------------------------
# Hard upper bound on how many keyword/expansion widget rows the node can
# show. The Python side always declares this many optional inputs; the JS
# frontend (web/dynamic_keyword_expander.js) shows/hides the widget rows
# beyond `field_count`. Outputs are NOT per-pair (see the class docstring
# below for why), so there's nothing dynamic to manage on the output side.
#
# Raise this if you need more than 20 pairs -- just keep the JS file's
# MAX_FIELDS constant in sync with this value.
# ---------------------------------------------------------------------------
MAX_FIELDS = 20


def _build_pattern(keyword: str, whole_word: bool):
    """Compile a case-insensitive regex for `keyword`.

    Whole-word matching uses \\b boundaries. If the keyword starts/ends with
    a non-word character, \\b won't behave intuitively there -- that's a
    known limitation of \\b-based whole-word matching in general.
    """
    escaped = re.escape(keyword)
    pattern = r"\b" + escaped + r"\b" if whole_word else escaped
    return re.compile(pattern, re.IGNORECASE)


def _process_pair(text: str, keyword: str, expansion: str, whole_word: bool, replace_all: bool):
    """Return (result_text, found) for a single keyword/expansion pair.

    - found is True iff the keyword occurs in `text` (respecting whole_word),
      even if `expansion` is an empty string.
    - result_text is `text` with the keyword occurrence(s) replaced by
      `expansion`. If replace_all is False, only the first occurrence is
      replaced. If the keyword isn't found, result_text == text unchanged.
    """
    if not keyword:
        return text, False

    pattern = _build_pattern(keyword, whole_word)
    found = pattern.search(text) is not None

    if not found:
        return text, False

    count = 0 if replace_all else 1
    # Use a lambda replacement so backslashes/group-refs in `expansion`
    # are treated as literal text, not regex replacement syntax.
    result = pattern.sub(lambda _m: expansion, text, count=count)
    return result, found


class DynamicKeywordExpander:
    """
    Dynamic Keyword Expander
    -------------------------
    Scans an input text against a configurable number of keyword/expansion
    pairs (rows 1..field_count, out of a max of MAX_FIELDS). For each active
    row:
      - the input text with occurrences of that row's keyword replaced by
        its expansion (case-insensitive; unmodified if not found).
      - whether that row's keyword occurs in the input text (True even if
        the expansion is an empty string).
      - the expansion value configured for that row (as-is, not substituted
        into anything).

    Plus one extra, non-list output:
      - `final`: the input text with ALL active rows' substitutions applied
        in sequence, one after another (row 1's substitution runs first,
        then row 2's runs on THAT result, and so on through field_count).
        This is different from `results`, where every row is computed
        independently against the original, unmodified `text`.

    `results`/`found`/`expansions` are emitted as LIST outputs (length
    `field_count`, in row order) rather than one set of sockets per row.
    This keeps the node's output signature fixed regardless of
    `field_count`, which sidesteps frontend issues with dynamically
    adding/removing per-instance output sockets. To pull out a single row's
    value, use the companion "List Get Item" node (also in this package)
    with the row's index, or feed the list into a node with
    INPUT_IS_LIST=True to work with the whole list at once.

    `field_count` also controls how many of the MAX_FIELDS keyword_i /
    expansion_i widget rows are shown in the UI (handled by the JS
    frontend) -- rows beyond `field_count` are ignored entirely here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "text": ("STRING", {"multiline": True, "default": ""}),
            "field_count": ("INT", {"default": 1, "min": 1, "max": MAX_FIELDS, "step": 1}),
            "whole_word": ("BOOLEAN", {"default": False}),
            "replace_all": ("BOOLEAN", {"default": True}),
        }

        optional = {}
        for i in range(1, MAX_FIELDS + 1):
            # Plain STRING (not forceInput) so ComfyUI renders a text widget
            # by default, while still letting the user right-click ->
            # "Convert widget to input" to feed it from another node.
            optional[f"keyword_{i}"] = ("STRING", {"default": "", "multiline": False})
            optional[f"expansion_{i}"] = ("STRING", {"default": "", "multiline": False})

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("results", "found", "expansions", "final")
    OUTPUT_IS_LIST = (True, True, True, False)

    FUNCTION = "expand"
    CATEGORY = "text/keyword_expander"

    def expand(self, text, field_count, whole_word, replace_all, **kwargs):
        field_count = max(1, min(MAX_FIELDS, int(field_count)))

        results = []
        found_flags = []
        expansions = []
        final_text = text
        for i in range(1, field_count + 1):
            keyword = kwargs.get(f"keyword_{i}", "") or ""
            expansion = kwargs.get(f"expansion_{i}", "") or ""

            # Independent per-row result, always against the original text.
            result, found = _process_pair(text, keyword, expansion, whole_word, replace_all)
            results.append(result)
            found_flags.append(found)
            if found:
                expansions.append(expansion)
            else:
                expansions.append("")

            # Cascaded result: each row's substitution builds on the last.
            final_text, _ = _process_pair(final_text, keyword, expansion, whole_word, replace_all)

        return (results, found_flags, expansions, final_text)


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


NODE_CLASS_MAPPINGS = {
    "DynamicKeywordExpander": DynamicKeywordExpander,
    "ListGetItem": ListGetItem,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicKeywordExpander": "Dynamic Keyword Expander",
    "ListGetItem": "List Get Item",
}