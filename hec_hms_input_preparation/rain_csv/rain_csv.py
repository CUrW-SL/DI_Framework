import csv
import pandas as pd
from datetime import datetime, timedelta

from algo_wrapper import Config, IOProcessor, Algorithm
from data_layer.base import get_engine, get_sessionmaker
from data_layer.timeseries import Timeseries


class RainCsvIO(IOProcessor):
    def get_input(self, **dynamic_args):
        tms_adapter = self.get_timeseries_adapter()
        loc_ids = self.input_config['location-ids']
        tms_dict = {}

        for loc_id in loc_ids:
            start_dt = dynamic_args['start_dt']
            base_dt = dynamic_args['base_dt']
            end_dt = dynamic_args['end_dt']

            obs_tms_meta = self.input_config['obs_tms_meta'][loc_id]
            obs_tms_id = tms_adapter.get_timeseries_id(obs_tms_meta)
            obs_tms = tms_adapter.get_timeseries(obs_tms_id, start_dt, base_dt)
            tms_dict[loc_id] = obs_tms

            frcst_tms_meta = self.input_config['forecast_tms_meta'][loc_id]
            frcst_delta = end_dt - base_dt
            frcst_days = frcst_delta.days + 1 if frcst_delta.seconds > 0 else frcst_delta.days
            frcst_start_dt = base_dt
            for day in range(0, frcst_days):
                o_day_template = 'Forecast-0-d'
                after_day_template = 'Forecast-{0}-d-after'
                event_type = o_day_template if day == 0 else after_day_template.format(day)
                frcst_tms_meta['event_type'] = event_type
                frcst_end_dt = frcst_start_dt + timedelta(days=1)
                frcst_end_dt = frcst_end_dt if end_dt > frcst_end_dt else end_dt
                frcst_tms_id = tms_adapter.get_timeseries_id(frcst_tms_meta)
                frcst_tms = tms_adapter.get_timeseries(frcst_tms_id, frcst_start_dt, frcst_end_dt)
                if loc_id not in tms_dict or tms_dict[loc_id] is None:
                    tms_dict[loc_id] = frcst_tms
                else:
                    tms_dict[loc_id] = tms_dict[loc_id].append(frcst_tms)
                frcst_start_dt = frcst_end_dt

        # Concat along column axis.
        tms_df = pd.concat(tms_dict, axis=1)
        # Flatten resulting column multi-index.
        tms_df.columns = tms_df.columns.get_level_values(0)
        tms_df.index.name = 'time'
        # TODO handle missing values
        return tms_df

    def push_output(self, algo_output, **dynamic_args):
        if algo_output is not None:
            with open(self.output_config['rain_csv_fp'], 'w') as out_f:
                csv_writer = csv.writer(out_f, delimiter=',', quotechar='|')
                csv_writer.writerows(algo_output)

    def get_timeseries_adapter(self):
        db_config = self.input_config['db_config']
        db_engine = get_engine(host=db_config['host'], port=db_config['port'], user=db_config['user'],
                               password=db_config['password'], db=db_config['db'])
        return Timeseries(get_sessionmaker(engine=db_engine))


class RainCsvAlgo(Algorithm):
    def algo(self, algo_input, **dynamic_args):
        location_ids = self.algo_config['location-ids']
        location_names = self.algo_config['location-names']

        try:
            self.check_input_integrity(algo_input)

            csv_rows = []
            loc_name_row = ['Location Names']
            loc_id_row = ['Location Ids']
            col_names_row = ['Time']

            loc_name_row.extend([loc_name for loc_name in location_names])
            loc_id_row.extend([loc_id for loc_id in location_ids])
            col_names_row.extend(['Rainfall' for _ in location_ids])
            csv_rows.extend([loc_name_row, loc_id_row, col_names_row])

            for index, row in algo_input.iterrows():
                ts = index.strftime(self.algo_config['timestamp-format'])
                values = [row[loc_id] for loc_id in location_ids]
                csv_row = [ts]
                csv_row.extend(values)
                csv_rows.append(csv_row)

            return csv_rows
        except AttributeError as ex:
            print('Input Integrity Error!', ex)
            return None

    def check_input_integrity(self, algo_input):
        location_ids = self.algo_config['location-ids']

        # Should be a pandas Dataframe with DatetimeIndex.
        if not isinstance(algo_input, pd.DataFrame) or not isinstance(algo_input.index, pd.DatetimeIndex):
            raise AttributeError('algo_input should be a pandas DataFrame with DatetimeIndex.')
        # Column names should be same as location_ids.
        if set(location_ids) != set(algo_input.columns):
            raise AttributeError('Columns values of input DataFrame should be same as location Ids.')
        return True


if __name__ == '__main__':
    raincsv_config = Config('/home/nira/PycharmProjects/DI_Framework/hec_hms_input_preparation/rain_csv/config.json')
    raincsv_io = RainCsvIO(raincsv_config)
    raincsv_algo = RainCsvAlgo(raincsv_io, raincsv_config)
    raincsv_algo.execute(
        start_dt=datetime(2018, 1, 1, 0, 0, 0),
        base_dt=datetime(2018, 1, 3, 0, 0, 0),
        end_dt=datetime(2018, 1, 6, 0, 0, 0),
    )
