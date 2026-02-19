import numpy as np
import xarray as xr
from scipy.special import roots_legendre
from .rotated_grid import rot_to_reg


default_attrs = {
    "lat": {
        "long_name": "latitude",
        "units": "degrees_north",
        "standard_name": "latitude",
    },
    "lon": {
        "long_name": "longitude",
        "units": "degrees_east",
        "standard_name": "longitude",
    },
    "time": {
        "units": "seconds since 1970-01-01T00:00:00",
        "calendar": "proleptic_gregorian",
    },
}


class GribGrid:
    _subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclasses.append(cls)


class GaussianReduced(GribGrid):
    gridType = "reduced_gg"
    params = ["pl"]

    @classmethod
    def compute_coords(cls, pl):
        lons = np.concatenate([np.linspace(0, 360, nl, endpoint=False) for nl in pl])
        single_lats = np.rad2deg(-np.arcsin(roots_legendre(len(pl))[0]))
        lats = np.concatenate([[lat] * nl for nl, lat in zip(pl, single_lats)])

        return xr.Dataset(
            coords={
                "lat": (("value",), lats, default_attrs["lat"]),
                "lon": (("value",), lons, default_attrs["lon"]),
            }
        )


class GaussianRegular(GribGrid):
    gridType = "regular_gg"
    params = ["N"]

    @classmethod
    def compute_coords(cls, N):
        lats = np.rad2deg(-np.arcsin(roots_legendre(2 * N)[0]))
        lons = np.linspace(0, 360, 4 * N, endpoint=False)

        return xr.Dataset(
            coords={
                "lat": (("lat",), lats, default_attrs["lat"]),
                "lon": (("lon",), lons, default_attrs["lon"]),
            }
        )


class LatLonReduced(GribGrid):
    gridType = "reduced_ll"
    params = ["pl"]

    @classmethod
    def compute_coords(cls, pl):
        lons = np.concatenate([np.linspace(0, 360, nl, endpoint=False) for nl in pl])
        single_lats = np.linspace(90, -90, len(pl), endpoint=True)
        lats = np.concatenate([[lat] * nl for nl, lat in zip(pl, single_lats)])

        return xr.Dataset(
            coords={
                "lat": (("value",), lats, default_attrs["lat"]),
                "lon": (("value",), lons, default_attrs["lon"]),
            }
        )


class LatLonRegular(GribGrid):
    gridType = "regular_ll"
    params = [
        "Ni",
        "Nj",
        "latitudeOfFirstGridPointInDegrees",
        "longitudeOfFirstGridPointInDegrees",
        "iDirectionIncrementInDegrees",
        "jDirectionIncrementInDegrees",
        "iScansNegatively",
        "jScansPositively",
    ]

    @classmethod
    def compute_coords(cls, **kwargs):
        Ni = kwargs["Ni"]
        Nj = kwargs["Nj"]
        iInc = kwargs["iDirectionIncrementInDegrees"]
        jInc = kwargs["jDirectionIncrementInDegrees"]
        lonFirst = kwargs["longitudeOfFirstGridPointInDegrees"]
        latFirst = kwargs["latitudeOfFirstGridPointInDegrees"]

        iInc = -iInc if kwargs["iScansNegatively"] else iInc
        jInc = jInc if kwargs["jScansPositively"] else -jInc

        lons = (lonFirst + np.arange(Ni) * iInc + 180) % 360 - 180
        lats = latFirst + np.arange(Nj) * jInc

        return xr.Dataset(
            coords={
                "lat": (("lat",), lats, default_attrs["lat"]),
                "lon": (("lon",), lons, default_attrs["lon"]),
            }
        )


class HEALPix(GribGrid):
    gridType = "healpix"
    params = [
        "orderingConvention",
        "Nside",
    ]

    @classmethod
    def compute_coords(cls, orderingConvention, Nside):
        import healpy as hp

        lons, lats = hp.pix2ang(
            nside=Nside,
            ipix=np.arange(hp.nside2npix(Nside)),
            nest=orderingConvention == "nested",
            lonlat=True,
        )

        return xr.Dataset(
            coords={
                "lat": (("value",), lats, default_attrs["lat"]),
                "lon": (("value",), lons, default_attrs["lon"]),
            }
        )


class SphericalHarmonics(GribGrid):
    gridType = "sh"
    params = []

    @classmethod
    def compute_coords(cls):
        return {}


class LatLonRotated(GribGrid):
    gridType = "rotated_ll"
    params = [
        "Ni",
        "Nj",
        "latitudeOfFirstGridPointInDegrees",
        "longitudeOfFirstGridPointInDegrees",
        "latitudeOfLastGridPointInDegrees",
        "longitudeOfLastGridPointInDegrees",
        "latitudeOfSouthernPoleInDegrees",
        "longitudeOfSouthernPoleInDegrees",
    ]

    @classmethod
    def compute_coords(cls, **kwargs):
        Ni = kwargs["Ni"]
        Nj = kwargs["Nj"]
        latFirst = kwargs["latitudeOfFirstGridPointInDegrees"]
        latLast = kwargs["latitudeOfLastGridPointInDegrees"]
        lonFirst = kwargs["longitudeOfFirstGridPointInDegrees"]
        lonLast = kwargs["longitudeOfLastGridPointInDegrees"]
        latPole = kwargs["latitudeOfSouthernPoleInDegrees"]
        lonPole = kwargs["longitudeOfSouthernPoleInDegrees"]

        lons, lats = np.meshgrid(
            np.linspace(lonFirst, lonLast, Ni), np.linspace(latFirst, latLast, Nj)
        )

        lons, lats = rot_to_reg(lonPole, latPole, lons, lats)

        # TODO: what should these coordinates actually be interpreted as? They
        # are not really "x" and "y" coordinates in a projection, but we need
        # some coordinates to index the data variables by
        x = np.linspace(0, 1, Ni)
        y = np.linspace(0, 1, Nj)
        
        return xr.Dataset(
            coords={
                "lat": (("y", "x"), lats, default_attrs["lat"]),
                "lon": (("y", "x"), lons, default_attrs["lon"]),
                "x": (("x",), x, {"long_name": "x coordinate", "units": "1", "standard_name": "projection_x_coordinate"}),
                "y": (("y",), y, {"long_name": "y coordinate", "units": "1", "standard_name": "projection_y_coordinate"}),
            }
        )
    
import functools
    

class Lambert(GribGrid):
    gridType = "lambert"
    params = [
        "Ni",
        "Nj",
        "LoVInDegrees",
        "LaDInDegrees",
        "Latin1InDegrees",
        "Latin2InDegrees",
        "latitudeOfFirstGridPointInDegrees",
        "longitudeOfFirstGridPointInDegrees",
        "iScansPositively",
        "jScansPositively",
        "DxInMetres",
        "DyInMetres",
        "shapeOfTheEarth",
        "radiusInMetres",
        "edition"
    ]

    @classmethod
    @functools.lru_cache(maxsize=128)
    def compute_coords(cls, **kwargs):
        Ni = kwargs["Ni"]
        Nj = kwargs["Nj"]
        lon_0 = kwargs["LoVInDegrees"] # Latitude of first standard parallel
        lat_0 = kwargs["LaDInDegrees"] # Latitude of second standard parallel
        lat_1 = kwargs["Latin1InDegrees"] # Origin latitude
        lat_2 = kwargs["Latin2InDegrees"] # Origin longitude
        lat1 = kwargs["latitudeOfFirstGridPointInDegrees"] # Origin latitude
        lon1 = kwargs["longitudeOfFirstGridPointInDegrees"] # Origin longitude
        iScansPositively = kwargs["iScansPositively"]
        jScansPositively = kwargs["jScansPositively"]
        dx = kwargs["DxInMetres"]
        dy = kwargs["DyInMetres"]
        shapeOfTheEarth = kwargs["shapeOfTheEarth"]
        edition = kwargs["edition"]
        
        # assume false_easting and false_northing are 0
        false_easting = 0
        false_northing = 0

        # a = Semi-major axis of reference ellipsoid
        # b = Semi-minor axis of reference ellipsoid
        if edition == 1:
            # for GRIB1 we ignore shapeOfTheEarth and instead use radiusInMetres
            # we do this because eccodes prior to 2.33.0 returned the incorrect value for shapeOfTheEarth
            # see https://jira.ecmwf.int/browse/ECC-811 and https://github.com/ecmwf/eccodes/commit/4808994174d735008e70c4b032d447369362ce92
            a = b = kwargs["radiusInMetres"]
        elif edition == 2:
            if shapeOfTheEarth == 6:
                a = b = 6371229.0 
            else:
                raise NotImplementedError("Only sphere implemented")
        else:
            raise NotImplementedError("Only GRIB1 and GRIB2 implemented")
        
        f = (a-b)/a # f = Flattening of reference ellipsoid

        if lat_1 == lat_2:
            # Lambert Conic Conformal (1SP)
            pass
        else:
            raise NotImplementedError("Lambert Conic Conformal (2SP) not implemented")

        phi = np.deg2rad(lat1)
        lambd = np.deg2rad(lon1)
        phi_0 = np.deg2rad(lat_0)
        theta_0 = np.deg2rad(lon_0)

        k0 = 1.000000 # scale factor at natural origin
        e = np.sqrt(2*f-f**2)

        m0 = np.cos(phi_0)/(1-e**2 * np.sin(phi_0)**2)**(1/2)
        t0 = np.tan(np.pi/4 - phi_0/2)/((1-e*np.sin(phi_0))/(1+e*np.sin(phi_0)))**(e/2)
        t = np.tan(np.pi/4 - phi/2)/((1-e*np.sin(phi))/(1+e*np.sin(phi)))**(e/2)
        n = np.sin(phi_0)
        F = m0/(n*t0**n)
        r = a*F*t**n * k0
        r0 = a*F*t0**n * k0
        theta = n*(lambd-theta_0)

        llcrnrx = false_easting + r*np.sin(theta)
        llcrnry = false_northing + r0 - r*np.cos(theta)

        if jScansPositively == 0 and dy > 0: dy = -dy
        if iScansPositively == 0 and dx > 0: dx = -dx

        easting_ = llcrnrx + dx*np.arange(Ni)
        northing_ = llcrnry + dy*np.arange(Nj)
        easting, northing = np.meshgrid(easting_, northing_)
        
        @np.vectorize
        def _calc_lambda_phi(east, north):
            theta_dot = np.arctan((east-false_easting)/(r0-(north-false_northing)))
            r_dot = ((east-false_easting)**2+(r0-(north-false_northing))**2)**0.5
            t_dot = (r_dot/(a*k0*F))**(1/n)

            delta_phi=10
            epsilon = 10e-10


            phi_g = np.pi/2 - 2*np.arctan(t_dot)

            while delta_phi > epsilon:
                phi = np.pi/2 - 2*np.arctan(t_dot * ( (1-e*np.sin(phi_g)) / (1+e*np.sin(phi_g)) )**(e/2) )
                delta_phi = abs(phi-phi_g)
                phi_g = phi

            lambd = theta_dot/n + theta_0
            return lambd, phi
        
        # lambd, phi = _calc_lambda_phi(easting, northing)
        lambd, phi = _calc_lambda_phi_numba(easting, northing, false_easting, false_northing, r0, theta_0, a, k0, F, n, e)
        
        lats = np.rad2deg(phi) 
        lons = np.rad2deg(lambd)
                
        x = easting_
        y = northing_
        
        return xr.Dataset(
            coords={
                "lat": (("y", "x"), lats, default_attrs["lat"]),
                "lon": (("y", "x"), lons, default_attrs["lon"]),
                "x": (("lon",), x, {"long_name": "x coordinate (easting)", "units": "m", "standard_name": "projection_x_coordinate"}),
                "y": (("lat",), y, {"long_name": "y coordinate (northing)", "units": "m", "standard_name": "projection_y_coordinate"}),
            }
        )

import numba

@np.vectorize
@numba.jit
def _calc_lambda_phi_numba(east, north, false_easting, false_northing, r0, theta_0, a, k0, F, n, e):
    theta_dot = np.arctan((east-false_easting)/(r0-(north-false_northing)))
    r_dot = ((east-false_easting)**2+(r0-(north-false_northing))**2)**0.5
    t_dot = (r_dot/(a*k0*F))**(1/n)

    delta_phi=10
    epsilon = 10e-10


    phi_g = np.pi/2 - 2*np.arctan(t_dot)

    while delta_phi > epsilon:
        phi = np.pi/2 - 2*np.arctan(t_dot * ( (1-e*np.sin(phi_g)) / (1+e*np.sin(phi_g)) )**(e/2) )
        delta_phi = abs(phi-phi_g)
        phi_g = phi

    lambd = theta_dot/n + theta_0
    return lambd, phi


grids = {g.gridType: g for g in GribGrid._subclasses}


def params_for_gridType(gridType):
    if gridType in grids:
        return grids[gridType].params
    else:
        return []


def varinfo2coords(varinfo):
    grid = grids[varinfo["attrs"]["gridType"]]
    return grid.compute_coords(**{k: varinfo["extra"][k] for k in grid.params})
