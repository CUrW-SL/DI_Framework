import pandas as pd
from datetime import datetime

from algo_wrapper import Config, IOProcessor, Algorithm
from data_layer.base import get_engine, get_sessionmaker
from data_layer.timeseries import Timeseries


class OutflowIO(IOProcessor):

    def get_input(self, **dynamic_args):
        print('retrieving input...')
        tms_adapter = self.get_timeseries_adapter()
        tidal_tms_id = tms_adapter.get_timeseries_id(meta_data=self.input_config['tidal-forecast-timeseries-meta'])
        tidal_tms = tms_adapter.get_timeseries(tidal_tms_id, dynamic_args['start_dt'], dynamic_args['end_dt'])
        return tidal_tms

    def push_output(self, algo_output, **dynamic_args):
        print('saving the output...')
        with open(self.output_config['outflow_dat_fp'], 'w') as out_f:
            out_f.writelines(algo_output)

    def get_timeseries_adapter(self):
        db_config = self.input_config['db_config']
        db_engine = get_engine(host=db_config['host'], port=db_config['port'], user=db_config['user'],
                               password=db_config['password'], db=db_config['db'])
        return Timeseries(get_sessionmaker(engine=db_engine))


class OutflowAlgo(Algorithm):

    def algo(self, algo_input, **dynamic_args):
        hourly_forecast = OutflowAlgo.process_tidal_forecast(algo_input)
        lines = []
        with open(self.algo_config['init_tidal_config']) as init_tidal_conf_f:
            init_tidal_levels = init_tidal_conf_f.readlines()
            for init_tidal_level in init_tidal_levels:
                if len(init_tidal_level.split()):  # Check if not empty line
                    lines.append(init_tidal_level)
                    if init_tidal_level[0] == 'N':
                        lines.append('{0} {1:{w}} {2:{w}}\n'.format('S', 0, 0, w=self.algo_config['DAT_WIDTH']))
                        base_dt = dynamic_args['start_dt'].replace(minute=0, second=0, microsecond=0)
                        for dt_index, rows in hourly_forecast.iterrows():
                            hours_so_far = int((dt_index - base_dt).total_seconds()/3600)
                            tidal_value = float(rows['value'])
                            tidal_line = '{0} {1:{w}} {2:{w}{b}}\n'\
                                .format('S', hours_so_far, tidal_value, b='.2f', w=self.algo_config['DAT_WIDTH'])
                            lines.append(tidal_line)
        return lines

    @staticmethod
    def process_tidal_forecast(tidal_forecast):
        if not isinstance(tidal_forecast, pd.DataFrame):
            raise TypeError('Given timeseries is not a pandas data-frame of time, value columns')
        return tidal_forecast.resample('H').max().dropna()


if __name__ == '__main__':
    outflow_config = Config('/home/nira/PycharmProjects/DI_Framework/flo2d_input_preparation/outflow/config.json')
    outflow_io = OutflowIO(outflow_config)
    outflow_algo = OutflowAlgo(outflow_io, outflow_config)
    outflow_algo.execute(start_dt=datetime(2018,1,1,0,0,0), end_dt=datetime(2018,1,5,0,0,0))
