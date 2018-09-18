from glob import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from algo_wrapper import Config, IOProcessor, Algorithm
from data_layer.base import get_engine, get_sessionmaker
from data_layer.timeseries import Timeseries
from di_framework.utils.wrf.extraction import ext_utils
from di_framework.utils import common_utils


class RaincellNcfIO(IOProcessor):
    def get_input(self, **dynamic_args):
        print('retrieving input...')

        points = np.genfromtxt(self.input_config['kelani_basin_cell_lon_lat_map_file'], delimiter=',')
        kel_lon_min = np.min(points, 0)[1]
        kel_lat_min = np.min(points, 0)[2]
        kel_lon_max = np.max(points, 0)[1]
        kel_lat_max = np.max(points, 0)[2]

        nc_f = dynamic_args['ncfs']['nc_f']
        diff, kel_lats, kel_lons, times = ext_utils.\
            extract_area_rf_series(nc_f, kel_lat_min, kel_lat_max, kel_lon_min, kel_lon_max)
        times = pd.to_datetime(times, format='%Y-%m-%d_%H:%M:%S')

        def get_bins(arr):
            sz = len(arr)
            return (arr[1:sz - 1] + arr[0:sz - 2]) / 2

        lat_bins = get_bins(kel_lats)
        lon_bins = get_bins(kel_lons)

        t0 = times[0]
        t1 = times[1]
        t_end = times[-1]

        # previous rainfalls, usually 2 previous days.
        nc_f_prev_days = dynamic_args['ncfs']['nc_f_prev_days']
        prev_diff = []
        prev_days = len(nc_f_prev_days)
        for i in range(prev_days):
            if nc_f_prev_days[i]:
                p_diff, _, _, _ = ext_utils\
                    .extract_area_rf_series(nc_f_prev_days[i], kel_lat_min, kel_lat_max, kel_lon_min, kel_lon_max)
                prev_diff.append(p_diff)
            else:
                prev_diff.append(None)

        # preparing input for the first line of RAINCELL.DAT
        res_mins = int((t1 - t0).total_seconds() / 60)
        data_hours = int(len(times) + prev_days * 24 * 60 / res_mins)
        start_ts = common_utils.datetime_utc_to_lk(t0 - timedelta(days=prev_days), shift_mins=30)
        end_ts = common_utils.datetime_utc_to_lk(t_end, shift_mins=30)

        # preparing the rainfall of each cell_no
        rainfall_ = {}
        for point in points:
            cell_no = int(point[0])
            values = []
            rf_x = np.digitize(point[1], lon_bins)
            rf_y = np.digitize(point[2], lat_bins)
            for d in range(prev_days):
                for t in range(int(24 * 60 / res_mins)):
                    if prev_diff[prev_days - 1 - d] is not None:
                        values.append(prev_diff[prev_days - 1 - d][t, rf_y, rf_x])
                    else:
                        values.append(0)
            rainfall_[cell_no] = values

        alpha = self.input_config['target_rf']/self.input_config['avg_basin_rf']
        for point in points:
            cell_no = int(point[0])
            values = []
            rf_x = np.digitize(point[1], lon_bins)
            rf_y = np.digitize(point[2], lat_bins)
            for t in range(len(times)):
                if t < int(24 * 60 / res_mins):
                    values.append(diff[t, rf_y, rf_x] * alpha)
                else:
                    values.append(diff[t, rf_y, rf_x])
            rainfall_[cell_no] = values

        # preapring pandas Dataframe of rainfall. index: datetime of times, cloumns: cell_nos as int
        rainfall_df = pd.DataFrame(data=rainfall_, index=times)

        # preparing the input that is ready to be fed into algo.
        # {'res_min': , 'data_hours':, 'start_ts':, 'end_ts':, 'rainfall': }
        return {
            'res_min': res_mins,
            'data_hours': data_hours,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'rainfall': rainfall_df
        }

    def push_output(self, algo_output, **dynamic_args):
        print('saving the output...')


class RaincellHybridIO(IOProcessor):
    def get_input(self, **dynamic_args):
        print('retrieving input...')
        return None

    def push_output(self, algo_output, **dynamic_args):
        print('saving the output...')

    def get_timeseries_adapter(self):
        db_config = self.input_config['db_config']
        db_engine = get_engine(host=db_config['host'], port=db_config['port'], user=db_config['user'],
                               password=db_config['password'], db=db_config['db'])
        return Timeseries(get_sessionmaker(engine=db_engine))


class RaincellObsIO(IOProcessor):
    def get_input(self, **dynamic_args):
        print('retrieving input...')
        return None

    def push_output(self, algo_output, **dynamic_args):
        print('saving the output...')

    def get_timeseries_adapter(self):
        db_config = self.input_config['db_config']
        db_engine = get_engine(host=db_config['host'], port=db_config['port'], user=db_config['user'],
                               password=db_config['password'], db=db_config['db'])
        return Timeseries(get_sessionmaker(engine=db_engine))


class RaincellAlgo(Algorithm):

    def algo(self, algo_input, **dynamic_args):
        return None


if __name__ == '__main__':
    raincell_config = Config('/home/nira/PycharmProjects/DI_Framework/flo2d_input_preparation/raincell/config.json')
    raincell_io = RaincellNcfIO(raincell_config)
    outflow_algo = RaincellAlgo(raincell_io, raincell_config)
    outflow_algo.execute(
        ncfs={'nc_f': "/home/nira/PycharmProjects/DI_Framework/resources/wrf_output/now/wrfout_d03_2018-01-03_18_00_00_rf",
              'nc_f_prev_days': [
                  "/home/nira/PycharmProjects/DI_Framework/resources/wrf_output/prev_1/wrfout_d03_2018-01-02_18_00_00_rf",
                  "/home/nira/PycharmProjects/DI_Framework/resources/wrf_output/prev_2/wrfout_d03_2018-01-01_18_00_00_rf"
              ]
              },
        start_dt=datetime(2018, 1, 1, 0, 0, 0),
        base_dt=datetime(2018, 1, 1, 0, 0, 0),
        end_dt=datetime(2018, 1, 1, 0, 0, 0),
    )


def extract_kelani_basin_rainfall_flo2d(nc_f, nc_f_prev_days, output_dir, avg_basin_rf=1.0, kelani_basin_file=None,
                                        target_rfs=None, output_prefix='RAINCELL'):
    """
    :param output_prefix:
    :param nc_f:
    :param nc_f_prev_days:
    :param output_dir:
    :param avg_basin_rf:
    :param kelani_basin_file:
    :param target_rfs:
    :return:
    """
    if target_rfs is None:
        target_rfs = [100, 150, 200, 250, 300]
    if kelani_basin_file is None:
        raise AttributeError('kalani_basin_file cannot be None. Should have a valid file path.')

    points = np.genfromtxt(kelani_basin_file, delimiter=',')

    kel_lon_min = np.min(points, 0)[1]
    kel_lat_min = np.min(points, 0)[2]
    kel_lon_max = np.max(points, 0)[1]
    kel_lat_max = np.max(points, 0)[2]

    diff, kel_lats, kel_lons, times = ext_utils.extract_area_rf_series(nc_f, kel_lat_min, kel_lat_max, kel_lon_min,
                                                                       kel_lon_max)

    def get_bins(arr):
        sz = len(arr)
        return (arr[1:sz - 1] + arr[0:sz - 2]) / 2

    lat_bins = get_bins(kel_lats)
    lon_bins = get_bins(kel_lons)

    t0 = datetime.strptime(times[0], '%Y-%m-%d_%H:%M:%S')
    t1 = datetime.strptime(times[1], '%Y-%m-%d_%H:%M:%S')
    t_end = datetime.strptime(times[-1], '%Y-%m-%d_%H:%M:%S')

    prev_diff = []
    prev_days = len(nc_f_prev_days)
    for i in range(prev_days):
        if nc_f_prev_days[i]:
            p_diff, _, _, _ = ext_utils.extract_area_rf_series(nc_f_prev_days[i], kel_lat_min, kel_lat_max, kel_lon_min,
                                                               kel_lon_max)
            prev_diff.append(p_diff)
        else:
            prev_diff.append(None)

    def write_forecast_to_raincell_file(output_file_path, alpha):
        with open(output_file_path, 'w') as output_file:
            res_mins = int((t1 - t0).total_seconds() / 60)
            data_hours = int(len(times) + prev_days * 24 * 60 / res_mins)
            start_ts = common_utils.datetime_utc_to_lk(t0 - timedelta(days=prev_days), shift_mins=30).strftime(
                '%Y-%m-%d %H:%M:%S')
            end_ts = common_utils.datetime_utc_to_lk(t_end, shift_mins=30).strftime('%Y-%m-%d %H:%M:%S')

            output_file.write("%d %d %s %s\n" % (res_mins, data_hours, start_ts, end_ts))

            for d in range(prev_days):
                for t in range(int(24 * 60 / res_mins)):
                    for point in points:
                        rf_x = np.digitize(point[1], lon_bins)
                        rf_y = np.digitize(point[2], lat_bins)
                        if prev_diff[prev_days - 1 - d] is not None:
                            output_file.write('%d %.1f\n' % (point[0], prev_diff[prev_days - 1 - d][t, rf_y, rf_x]))
                        else:
                            output_file.write('%d %.1f\n' % (point[0], 0))

            for t in range(len(times)):
                for point in points:
                    rf_x = np.digitize(point[1], lon_bins)
                    rf_y = np.digitize(point[2], lat_bins)
                    if t < int(24 * 60 / res_mins):
                        output_file.write('%d %.1f\n' % (point[0], diff[t, rf_y, rf_x] * alpha))
                    else:
                        output_file.write('%d %.1f\n' % (point[0], diff[t, rf_y, rf_x]))
