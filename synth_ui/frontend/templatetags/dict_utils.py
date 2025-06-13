"""Template filters for dictionary utilities in Django templates.

This module provides a filter to safely retrieve dict items in templates.
"""

from django import template

register = template.Library()


@register.filter(name="get_item")
def get_item(dictionary, key):
    """Safely get an item from a dictionary in templates.

    Parameters
    ----------
    dictionary : dict-like
        The dictionary or mapping to retrieve from.
    key : Any
        The key for which to get the value.

    Returns
    -------
    Any
        The value of dictionary[key] if present; otherwise an empty string.
    """
    if hasattr(dictionary, "get") and callable(
        dictionary.get
    ):  # Check if it's dict-like
        return dictionary.get(key, "")
    return ""  # Return empty string for non-dict types or if key is missing
