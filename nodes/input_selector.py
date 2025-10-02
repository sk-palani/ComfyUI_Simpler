"""
Input Selector node for ComfyUI
Selects between two inputs based on a boolean condition and displays which input was selected
"""

from server import PromptServer


# This special class helps with wildcard inputs and outputs
class AnyType(str):
    """
    Special class to denote that this node can output any type.
    Used for wildcard connections in ComfyUI.
    """

    def __ne__(self, __value: object) -> bool:
        return False


# Register the special type
ANY_TYPE = AnyType("*")


class InputSelector:
    """
    A utility node that selects between two inputs based on a boolean condition.
    Displays which input was selected using the progress text area.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Use STRING here instead of wildcard to ensure compatibility
                "condition": ("BOOLEAN", {"default": True}),
                # Boolean to control which input to select
            },
            # Use optional inputs to allow any type
            "optional": {
                "when_true": (ANY_TYPE, {}),  # Wildcard input for TRUE condition
                "when_false": (ANY_TYPE, {}),  # Wildcard input for FALSE condition
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (ANY_TYPE,)  # Use our special type for the output
    RETURN_NAMES = ("selected",)
    FUNCTION = "select_input"
    CATEGORY = "utils"

    # This is essential for any-to-any nodes
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    # Allow connecting any node outputs to our inputs
    def INPUTS_MATCH(self, when_true=None, when_false=None, **kwargs):
        return True

    def select_input(self, condition, when_true=None, when_false=None, unique_id=None):
        """
        Select between when_true and when_false based on the condition.
        Shows the selected input path in the progress text area.

        Args:
            condition: Boolean determining which input to select
            when_true: Value to return if condition is True
            when_false: Value to return if condition is False
            unique_id: Unique identifier for this node instance

        Returns:
            Tuple containing the selected input value
        """
        if condition:
            message = '<div style="background-color: #00cc00; padding: 4px 8px; border-radius: 3px; color: white; width: 100%; text-align: center; box-sizing: border-box;">TRUE</div>'
            if unique_id is not None:
                PromptServer.instance.send_progress_text(message, unique_id)
            return (when_true,)
        else:
            message = '<div style="background-color: #cc0000; padding: 4px 8px; border-radius: 3px; color: white; width: 100%; text-align: center; box-sizing: border-box;">FALSE</div>'
            if unique_id is not None:
                PromptServer.instance.send_progress_text(message, unique_id)
            return (when_false,)


# Node registration dictionary
NODE_CLASS_MAPPINGS = {
    "InputSelector": InputSelector,
}

# Optional display names for nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "InputSelector": "Input Selector",
}
