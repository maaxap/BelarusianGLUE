import builtins
import json
import re

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("lowercase")
class LowercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.lower() for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("uppercase")
class UppercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.upper() for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("map")
class MapFilter(Filter):
    def __init__(self, mapping_dict: dict = None, default_value=None) -> None:
        """
        Initializes the MapFilter with a given mapping dictionary and default value.

        Args:
        - mapping_dict (dict): A dictionary containing the key-value mappings.
                               Default is an empty dictionary.
        - default_value (Any): The value to be returned when a key is not found in the mapping_dict.
                               Default is None.

        Example:
        mapper = MapFilter({'A': 1, 'B': 2}, default_value=0)
        """
        if mapping_dict is None:
            mapping_dict = {}
        assert isinstance(
            mapping_dict, dict
        ), "Provided mapping_dict is not a dictionary"
        self.mapping_dict = mapping_dict
        self.default_value = default_value

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp, self.default_value) for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("cast_to_dtype")
class CastToDtypeFilter(Filter):
    def __init__(self, dtype: str) -> None:
        """
        Initializes the CastDtypeFilter with a given data type.

        Args:
        - dtype (str): The data type to cast the responses to.
                       Must be a valid built-in Python data type.

        Example:
        caster = CastDtypeFilter('int')
        """
        super().__init__()

        assert dtype in {
            "int",
            "float",
            "str",
            "bool",
            "complex",
            "bytes",
            "bytearray",
            "memoryview",
            "list",
            "tuple",
            "set",
            "frozenset",
            "dict",
        }, f"Provided dtype is not a valid built-in Python data type: {dtype}"

        self.dtype = getattr(builtins, dtype, None)

    def apply(self, resps, docs):
        def cast_to_dtype(inst):
            return [self.dtype(resp) for resp in inst]

        return [cast_to_dtype(resp) for resp in resps]


@register_filter("parse_json_markdown")
class ParseJsonMarkdownFilter(Filter):
    JSON_MARKDOWN_PATTERN = re.compile(r"```(json)?(.*)```", re.DOTALL)

    @staticmethod
    def _parse_json_markdown(json_string):
        # The implementation is borrowed from
        #   https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/utils/json.py#L124
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            # Try to find JSON string within triple backticks
            match = ParseJsonMarkdownFilter.JSON_MARKDOWN_PATTERN.search(json_string)

            # If no match found, raise error
            if match is None:
                raise e

            # If match found, use the content within the backticks
            json_string = match.group(2)
            json_string = json_string.strip()

            return json.loads(json_string)

    def apply(self, resps, docs):
        def parse_json(inst):
            return [self._parse_json_markdown(resp) for resp in inst]

        return [parse_json(resp) for resp in resps]
