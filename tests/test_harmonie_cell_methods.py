from gribscan.magician import Grib2TimeStat
from gribscan.magician import HarmonieMagician
from gribscan.magician import grib2_to_cf_cell_methods


def test_grib2_to_cf_cell_methods_mean_with_interval():
    meta = Grib2TimeStat(
        type_of_statistical_processing=0,
        length_of_time_range=6,
        indicator_of_unit_of_time_range=1,
    )
    assert grib2_to_cf_cell_methods(meta) == "time: mean (interval: 6 hour)"


def test_grib2_to_cf_cell_methods_point_for_instantaneous():
    meta = Grib2TimeStat(
        type_of_statistical_processing=255,
        length_of_time_range=0,
        indicator_of_unit_of_time_range=1,
    )
    assert grib2_to_cf_cell_methods(meta) == "time: point"


def test_harmonie_magician_sets_cell_methods_for_statistical_fields():
    magician = HarmonieMagician()
    info = {
        "dims": ("posix_time", "level"),
        "attrs": {"shortName": "tp"},
        "extra": {
            "typeOfStatisticalProcessing": 1,
            "lengthOfTimeRange": 3,
            "indicatorOfUnitOfTimeRange": 1,
        },
    }

    result = magician.variable_hook(("tp", "surface"), info)
    assert result["name"] == "tp"
    assert result["attrs"]["cell_methods"] == "time: sum (interval: 3 hour)"
    assert result["attrs"]["coordinates"] == "lon lat"
    assert result["dims"] == ("time", "level")


def test_harmonie_magician_skips_cell_methods_for_unknown_with_interval():
    magician = HarmonieMagician()
    info = {
        "dims": ("posix_time",),
        "attrs": {"shortName": "unknown"},
        "extra": {
            "typeOfStatisticalProcessing": 999,
            "lengthOfTimeRange": 6,
            "indicatorOfUnitOfTimeRange": 1,
        },
    }

    result = magician.variable_hook(("foo", "surface"), info)
    assert result["name"] == "foo"
    assert "cell_methods" not in result["attrs"]
