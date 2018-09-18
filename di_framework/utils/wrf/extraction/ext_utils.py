import os
import datetime as dt

import numpy as np
from numpy.lib.recfunctions import append_fields
from netCDF4 import Dataset

from di_framework.utils import common_utils


def extract_time_data(nc_f):
    nc_fid = Dataset(nc_f, 'r')
    times_len = len(nc_fid.dimensions['Time'])
    try:
        times = [''.join(x) for x in nc_fid.variables['Times'][0:times_len]]
    except TypeError:
        times = np.array([''.join([y.decode() for y in x]) for x in nc_fid.variables['Times'][:]])
    nc_fid.close()
    return times_len, times


def get_two_element_average(prcp, return_diff=True):
    avg_prcp = (prcp[1:] + prcp[:-1]) * 0.5
    if return_diff:
        return avg_prcp - np.insert(avg_prcp[:-1], 0, [0], axis=0)
    else:
        return avg_prcp


def extract_points_array_rf_series(nc_f, points_array, boundaries=None, rf_var_list=None, lat_var='XLAT',
                                   lon_var='XLONG', time_var='Times'):
    """
    :param boundaries: list [lat_min, lat_max, lon_min, lon_max]
    :param nc_f:
    :param points_array: multi dim array (np structured array)  with a row [name, lon, lat]
    :param rf_var_list:
    :param lat_var:
    :param lon_var:
    :param time_var:
    :return: np structured array with [(time, name1, name2, .... )]
    """

    if rf_var_list is None:
        rf_var_list = ['RAINC', 'RAINNC']

    if boundaries is None:
        lat_min = np.min(points_array[points_array.dtype.names[2]])
        lat_max = np.max(points_array[points_array.dtype.names[2]])
        lon_min = np.min(points_array[points_array.dtype.names[1]])
        lon_max = np.max(points_array[points_array.dtype.names[1]])
    else:
        lat_min, lat_max, lon_min, lon_max = boundaries

    variables = extract_variables(nc_f, rf_var_list, lat_min, lat_max, lon_min, lon_max, lat_var, lon_var, time_var)

    prcp = variables[rf_var_list[0]]
    for i in range(1, len(rf_var_list)):
        prcp = prcp + variables[rf_var_list[i]]

    diff = get_two_element_average(prcp, return_diff=True)

    result = np.array([common_utils.datetime_utc_to_lk(dt.datetime.strptime(t, '%Y-%m-%d_%H:%M:%S'), shift_mins=30).strftime(
        '%Y-%m-%d %H:%M:%S').encode('utf-8') for t in variables[time_var][:-1]], dtype=np.dtype([(time_var, 'U19')]))

    for p in points_array:
        lat_start_idx = np.argmin(abs(variables['XLAT'] - p[2]))
        lon_start_idx = np.argmin(abs(variables['XLONG'] - p[1]))
        rf = np.round(diff[:, lat_start_idx, lon_start_idx], 6)
        # use this for 4 point average
        # rf = np.round(np.mean(diff[:, lat_start_idx:lat_start_idx + 2, lon_start_idx:lon_start_idx + 2], axis=(1, 2)),
        #               6)
        result = append_fields(result, p[0].decode(), rf, usemask=False)

    return result


def extract_variables(nc_f, var_list, lat_min, lat_max, lon_min, lon_max, lat_var='XLAT', lon_var='XLONG',
                      time_var='Times'):
    """
    extract variables from a netcdf file
    :param nc_f: 
    :param var_list: comma separated string for variables / list of strings 
    :param lat_min: 
    :param lat_max: 
    :param lon_min: 
    :param lon_max: 
    :param lat_var: 
    :param lon_var: 
    :param time_var: 
    :return: 
    variables dict {var_key --> var[time, lat, lon], xlat --> [lat], xlong --> [lon], times --> [time]}
    """
    if not os.path.exists(nc_f):
        raise IOError('File %s not found' % nc_f)

    nc_fid = Dataset(nc_f, 'r')

    times = np.array([''.join([y.decode() for y in x]) for x in nc_fid.variables[time_var][:]])
    lats = nc_fid.variables[lat_var][0, :, 0]
    lons = nc_fid.variables[lon_var][0, 0, :]

    lat_inds = np.where((lats >= lat_min) & (lats <= lat_max))
    lon_inds = np.where((lons >= lon_min) & (lons <= lon_max))

    vars_dict = {}
    if isinstance(var_list, str):
        var_list = var_list.replace(',', ' ').split()
    # var_list = var_list.replace(',', ' ').split() if isinstance(var_list, str) else var_list
    for var in var_list:
        vars_dict[var] = nc_fid.variables[var][:, lat_inds[0], lon_inds[0]]

    nc_fid.close()

    vars_dict[time_var] = times
    vars_dict[lat_var] = lats[lat_inds[0]]
    vars_dict[lon_var] = lons[lon_inds[0]]

    return vars_dict


def get_mean_cell_size(lats, lons):
    return np.round(np.mean(np.append(lons[1:len(lons)] - lons[0: len(lons) - 1], lats[1:len(lats)]
                                      - lats[0: len(lats) - 1])), 3)


def extract_area_rf_series(nc_f, lat_min, lat_max, lon_min, lon_max):
    if not os.path.exists(nc_f):
        raise IOError('File %s not found' % nc_f)

    nc_fid = Dataset(nc_f, 'r')

    times_len, times = extract_time_data(nc_f)
    lats = nc_fid.variables['XLAT'][0, :, 0]
    lons = nc_fid.variables['XLONG'][0, 0, :]

    lon_min_idx = np.argmax(lons >= lon_min) - 1
    lat_min_idx = np.argmax(lats >= lat_min) - 1
    lon_max_idx = np.argmax(lons >= lon_max)
    lat_max_idx = np.argmax(lats >= lat_max)

    prcp = nc_fid.variables['RAINC'][:, lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx] + nc_fid.variables['RAINNC'][
                                                                                            :, lat_min_idx:lat_max_idx,
                                                                                            lon_min_idx:lon_max_idx]

    diff = get_two_element_average(prcp)

    nc_fid.close()

    return diff, lats[lat_min_idx:lat_max_idx], lons[lon_min_idx:lon_max_idx], np.array(times[0:times_len - 1])
