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

        # converting to DatetimeIndex.
        times = pd.to_datetime(times, format='%Y-%m-%d_%H:%M:%S')
        # converting utc time to lk time shifting by 6hs. utc to lk shift: 5h 30min + additional shift: 30min.
        times = times.shift(1, freq=timedelta(hours=6))

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
        start_ts = t0 - timedelta(days=prev_days)
        end_ts = t_end

        # preparing the rainfall of each cell_no
        prev_times_len = int(24 * 60 / res_mins)
        rainfall_ = {}
        for point in points:
            cell_no = int(point[0])
            rf_x = np.digitize(point[1], lon_bins)
            rf_y = np.digitize(point[2], lat_bins)
            prev_values = []
            for d in range(prev_days):
                for t in range(prev_times_len):
                    if prev_diff[prev_days - 1 - d] is not None:
                        prev_values.append(prev_diff[prev_days - 1 - d][t, rf_y, rf_x])
                    else:
                        prev_values.append(0)
            rainfall_[cell_no] = prev_values
            del prev_values

        alpha = self.input_config['target_rf']/self.input_config['avg_basin_rf']
        for point in points:
            cell_no = int(point[0])
            rf_x = np.digitize(point[1], lon_bins)
            rf_y = np.digitize(point[2], lat_bins)
            values = []
            for t in range(len(times)):
                if t < int(24 * 60 / res_mins):
                    values.append(diff[t, rf_y, rf_x] * alpha)
                else:
                    values.append(diff[t, rf_y, rf_x])
            rainfall_[cell_no].extend(values)
            del values

        # preapring pandas Dataframe of rainfall. index: datetime of times, cloumns: cell_nos as int
        prev_times = pd.DatetimeIndex(start=start_ts, freq=timedelta(minutes=res_mins),
                                      periods=prev_times_len*prev_days,)
        time_index = prev_times.append(times)
        rainfall_df = pd.DataFrame(data=rainfall_, index=time_index)
        del prev_times, times, rainfall_, time_index

        # preparing the input that is ready to be fed into algo.
        # {'res_min': , 'data_batch_size':, 'start_ts':, 'end_ts':, 'rainfall': }
        return {
            'res_min': res_mins,
            'batch_size': data_hours,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'rainfall': rainfall_df
        }

    def push_output(self, algo_output, **dynamic_args):
        print('saving the output...')
        if algo_output is not None:
            with open(self.output_config['outflow_dat_fp'], 'w') as out_f:
                out_f.writelines(algo_output)


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
        try:
            RaincellAlgo.check_input_integrity(algo_input)
            return RaincellAlgo.prepare_line(res_min=algo_input['res_min'],
                                             batch_size=algo_input['batch_size'],
                                             start_ts=algo_input['start_ts'],
                                             end_ts=algo_input['end_ts'],
                                             rainfall_df=algo_input['rainfall'])
        except AttributeError as ex:
            print('Input Integrity Error: ', ex)
            return None

    @staticmethod
    def check_input_integrity(algo_input):
        # Check the integrity of key.
        input_keys = ['res_min', 'batch_size', 'start_ts', 'end_ts', 'rainfall']
        if not algo_input.keys() == set(input_keys):
            raise AttributeError('Invalid input. algo_input should be a dict with the keys: (%s)' % input_keys)
        rainfall_time_index = algo_input['rainfall'].index
        if not algo_input['batch_size'] == rainfall_time_index.size:
            raise AttributeError('batch_size and size of time-index of rainfall data-frame should be the same.')
        if not rainfall_time_index[0] == algo_input['start_ts']:
            raise AttributeError('Start timestamp (start_ts) and the first value of rainfall-time-index is different.')
        if not rainfall_time_index[-1] == algo_input['end_ts']:
            raise AttributeError('End timestamp (end_ts) and the last value of rainfall-time-index is different.')
        return True

    @staticmethod
    def prepare_line(res_min, batch_size, start_ts, end_ts, rainfall_df):
        lines = []
        header_line = "%d %d %s %s\n" % (res_min, batch_size, start_ts, end_ts)
        lines.append(header_line)
        cell_nos = np.sort(rainfall_df.columns)
        for index, row in rainfall_df.iterrows():
            for cell in cell_nos:
                line = "%d %.1f\n" % (cell, row[cell])
                lines.append(line)
        return lines


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
