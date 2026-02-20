from collections import OrderedDict

from gribscan.gribscan import _build_param_name
from gribscan.gribscan import _is_accumulated
from gribscan.gribscan import _to_scalar


def test_to_scalar_uses_last_element_for_list():
    assert _to_scalar([255, 1]) == 1


def test_is_accumulated_true_for_statistical_processing_code_1():
    assert _is_accumulated(1)
    assert _is_accumulated([255, 1])


def test_is_accumulated_false_for_non_accumulated_codes():
    assert not _is_accumulated(None)
    assert not _is_accumulated(0)
    assert not _is_accumulated([0, 255])


def test_build_param_name_keeps_shortname_for_non_accumulated():
    parameter_code = OrderedDict(
        [("discipline", 0), ("parameterCategory", 1), ("parameterNumber", 8)]
    )
    assert _build_param_name("tp", parameter_code, 0) == "tp"


def test_build_param_name_uses_parameter_code_for_unknown_shortname():
    parameter_code = OrderedDict(
        [("discipline", 0), ("parameterCategory", 1), ("parameterNumber", 8)]
    )
    assert _build_param_name("unknown", parameter_code, 0) == "0.1.8"


def test_build_param_name_appends_accum_suffix():
    parameter_code = OrderedDict(
        [("discipline", 0), ("parameterCategory", 1), ("parameterNumber", 8)]
    )
    assert _build_param_name("tp", parameter_code, 1) == "tp_accum"
    assert _build_param_name("unknown", parameter_code, 1) == "0.1.8_accum"
