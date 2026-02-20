from dataclasses import dataclass
from typing import Optional

import numpy as np
import numcodecs

from .gridutils import varinfo2coords


@dataclass
class Grib2TimeStat:
    # GRIB2 Code Table 4.10
    type_of_statistical_processing: Optional[int]
    # GRIB2 time range fields
    length_of_time_range: Optional[int]
    indicator_of_unit_of_time_range: Optional[int]  # Code Table 4.4


_GRIB_TIME_UNITS = {
    0: "minute",
    1: "hour",
    2: "day",
    3: "month",
    4: "year",
    5: "decade",
    6: "normal",
    7: "century",
    10: "3 hours",
    11: "6 hours",
    12: "12 hours",
    13: "second",
}

_GRIB_STAT_TO_CF = {
    0: "mean",
    1: "sum",
    2: "maximum",
    3: "minimum",
    4: "difference",
    5: "rms",
    6: "std",
    7: "covariance",
    8: "difference",  # absolute difference
    9: "ratio",
    10: "standard_deviation",
    255: None,  # missing
}


def _to_scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return int(value[-1])
    return value


def _format_interval(length: Optional[int], unit_code: Optional[int]) -> Optional[str]:
    if length is None or unit_code is None:
        return None
    unit = _GRIB_TIME_UNITS.get(unit_code, f"unit_{unit_code}")
    return f"{length} {unit}"


def grib2_to_cf_cell_methods(meta: Grib2TimeStat) -> Optional[str]:
    verb = _GRIB_STAT_TO_CF.get(meta.type_of_statistical_processing)
    interval = _format_interval(
        meta.length_of_time_range, meta.indicator_of_unit_of_time_range
    )

    if verb is None:
        # No statistical processing. If no interval or zero interval, treat as instantaneous.
        if interval is None or meta.length_of_time_range == 0:
            return "time: point"
        return None

    return f"time: {verb} (interval: {interval})" if interval else f"time: {verb}"


class MagicianBase:
    def variable_hook(self, key, info):
        ...

    def globals_hook(self, global_attrs):
        return global_attrs

    def coords_hook(self, name, coords):
        # return {}, coords, {}, [name], None
        return coords, {}, None

    def m2key(self, meta):
        return tuple(meta[key] for key in self.varkeys), tuple(
            meta[key] for key in self.dimkeys
        )

    def m2dataset(self, meta):
        return (
            "atm3d"
            if meta["attrs"]["typeOfLevel"].startswith("generalVertical")
            else "atm2d"
        )

    def extra_coords(self, varinfo):
        return {}


class Magician(MagicianBase):
    varkeys = "param", "levtype"
    dimkeys = "posix_time", "level"

    def globals_hook(self, global_attrs):
        history = global_attrs.get("history", "")
        if len(history) > 0 and not history.endswith("\n"):
            history = history + "\r\n"
        history += "🪄🧙‍♂️🔮 magic dataset assembly provided by gribscan.Magician\r\n"
        return {**global_attrs, "history": history}

    def variable_hook(self, key, info):
        param, levtype = key
        name = param
        dims = info["dims"]

        if levtype == "generalVertical":
            name = param + "half" if param == "zg" else param
            dims = tuple("halflevel" if dim == "level" else dim for dim in dims)
        if levtype == "generalVerticalLayer":
            dims = tuple("fulllevel" if dim == "level" else dim for dim in dims)
        dims = tuple("time" if dim == "posix_time" else dim for dim in dims)

        return {
            "dims": dims,
            "name": name,
        }

    def coords_hook(self, name, coords):
        return coords, {}, None


class IFSMagician(MagicianBase):
    varkeys = "param", "levtype"
    dimkeys = "posix_time", "level"

    def globals_hook(self, global_attrs):
        history = global_attrs.get("history", "")
        if len(history) > 0 and not history.endswith("\n"):
            history = history + "\r\n"
        history += "🪄🧙‍♂️🔮 magic dataset assembly provided by gribscan.IFSMagician\r\n"
        return {**global_attrs, "history": history}

    def variable_hook(self, key, info):
        param, levtype = key
        name = param
        dims = info["dims"]

        if levtype == "generalVertical":
            name = param + "half" if param == "zg" else param
            dims = tuple("halflevel" if dim == "level" else dim for dim in dims)
        if levtype == "generalVerticalLayer":
            dims = tuple("fulllevel" if dim == "level" else dim for dim in dims)
        dims = tuple("time" if dim == "posix_time" else dim for dim in dims)

        return {
            "dims": dims,
            "name": name,
            "attrs": {
                **info["attrs"],
                "coordinates": "lon lat",
                "missingValue": 9999,
            },
        }

    def coords_hook(self, name, coords):
        compressor = numcodecs.Blosc("zstd")
        return coords, {}, compressor

    def m2dataset(self, meta):
        return (
            "atm3d"
            if meta["attrs"]["typeOfLevel"].startswith("isobaricInhPa")
            else "atm2d"
        )


class EnsembleMagician(IFSMagician):
    varkeys = "param", "levtype"
    dimkeys = "posix_time", "level", "member"

    def m2dataset(self, meta):
        """Divide datasets based on the IFS ensemble products description.

        Reference:
          https://www.ecmwf.int/en/forecasts/datasets/open-data#ensemble-products
        """
        if meta["member"] is None:
            if meta["attrs"]["shortName"] in ("gh", "t", "ws", "msl"):
                return "ensmean"
            else:
                return "prob"
        if meta["attrs"]["typeOfLevel"].startswith("isobaricInhPa"):
            return "atm3d"
        return "atm2d"


class HarmonieMagician(MagicianBase):
    varkeys = "param", "levtype"
    dimkeys = "posix_time", "level"

    def globals_hook(self, global_attrs):
        history = global_attrs.get("history", "")
        if len(history) > 0 and not history.endswith("\n"):
            history = history + "\r\n"
        history += (
            "🪄🧙‍♂️🔮 magic dataset assembly provided by gribscan.HarmonieMagician\r\n"
        )
        return {**global_attrs, "history": history}

    def variable_hook(self, key, info):
        param, levtype = key
        name = param
        dims = info["dims"]
        extra = info.get("extra", {})
        type_of_statistical_processing = _to_scalar(
            extra.get("typeOfStatisticalProcessing")
        )

        if levtype == "generalVertical":
            name = param + "half" if param == "zg" else param
            dims = tuple("halflevel" if dim == "level" else dim for dim in dims)
        if levtype == "generalVerticalLayer":
            dims = tuple("fulllevel" if dim == "level" else dim for dim in dims)
        dims = tuple("time" if dim == "posix_time" else dim for dim in dims)

        cell_methods = grib2_to_cf_cell_methods(
            Grib2TimeStat(
                type_of_statistical_processing=type_of_statistical_processing,
                length_of_time_range=_to_scalar(extra.get("lengthOfTimeRange")),
                indicator_of_unit_of_time_range=_to_scalar(
                    extra.get("indicatorOfUnitOfTimeRange")
                ),
            )
        )

        attrs = {
            **info["attrs"],
            "coordinates": "lon lat",
        }
        if cell_methods is not None:
            attrs["cell_methods"] = cell_methods

        return {
            "dims": dims,
            "data_dims": ["y", "x"],
            "data_shape": "__from_data_dims__",
            "name": name,
            "attrs": attrs,
        }

    def coords_hook(self, name, coords):
        attrs = {}
        compressor = numcodecs.Blosc("zstd")
        return coords, attrs, compressor

    def extra_coords(self, varinfo):
        v0 = next(iter(varinfo.values()))
        return varinfo2coords(v0)

    def m2dataset(self, meta):
        return meta["attrs"]["typeOfLevel"]


MAGICIANS = {
    "monsoon": Magician,
    "ifs": IFSMagician,
    "enfo": EnsembleMagician,
    "harmonie": HarmonieMagician,
}
