import pandas as pd
from datetime import datetime, timedelta

from algo_wrapper import Config, IOProcessor, Algorithm
from data_layer.base import get_engine, get_sessionmaker
from data_layer.timeseries import Timeseries


class InflowIO(IOProcessor):
    def get_input(self, **dynamic_args):
        print('retrieving input...')
        tms_adapter = self.get_timeseries_adapter()
        tms_dfs = []

        inflow_tms_meta = self.input_config['inflow-forecast-tms-meta']

        # From start_date to base_date get the discharge from Forecast-0-d timeseries.
        inflow_tms_meta['event_type'] = 'Forecast-0-d'
        print(inflow_tms_meta['event_type'])
        till_basedt_tms_id = tms_adapter.get_timeseries_id(meta_data=inflow_tms_meta)
        till_basedt_tms = tms_adapter.get_timeseries(till_basedt_tms_id,
                                                     dynamic_args['start_dt'], dynamic_args['base_dt'])
        print("From: ", dynamic_args['start_dt'])
        print("To: ", dynamic_args['base_dt'])
        tms_dfs.append(till_basedt_tms)

        # From base_date onwards get the discharge respectively from Forecast-1-d-after and respectively.
        end_base_delta = (dynamic_args['end_dt'] - dynamic_args['base_dt'])
        no_of_days = end_base_delta.days + 1 if end_base_delta.seconds > 0 else end_base_delta.days
        base_dt = dynamic_args['base_dt']
        for i in range(0, no_of_days):
            inflow_tms_meta['event_type'] = 'Forecast-{0}-d-after'.format(i+1)
            print(inflow_tms_meta['event_type'])
            after_basedt_tms_id = tms_adapter.get_timeseries_id(meta_data=inflow_tms_meta)
            start_dt = base_dt
            end_dt = start_dt + timedelta(days=1)
            # end_dt cannot go beyond given dynamic_args['start_dt']
            end_dt = end_dt if dynamic_args['end_dt'] >= end_dt else dynamic_args['end_dt']
            print("From: ", start_dt)
            print("To: ", end_dt)
            after_basedt_tms = tms_adapter.get_timeseries(after_basedt_tms_id, start_dt, end_dt)
            tms_dfs.append(after_basedt_tms)
            base_dt = end_dt
        return pd.concat(tms_dfs)

    def push_output(self, algo_output, **dynamic_args):
        print("pushing output....")
        with open(self.output_config['inflow_dat_fp'], 'w') as out_f:
            out_f.writelines(algo_output)

    def get_timeseries_adapter(self):
        db_config = self.input_config['db_config']
        db_engine = get_engine(host=db_config['host'], port=db_config['port'], user=db_config['user'],
                               password=db_config['password'], db=db_config['db'])
        return Timeseries(get_sessionmaker(engine=db_engine))


class InflowAlgo(Algorithm):

    def algo(self, algo_input, **dynamic_args):
        print("processing input....")
        lines = []

        line1 = '{0} {1:{w}{b}}\n'.format(self.algo_config['IHOURDAILY'], self.algo_config['IDEPLT'], b='d',
                                          w=self.algo_config['DAT_WIDTH'])
        line2 = '{0} {1:{w}{b}} {2:{w}{b}}\n'.format(self.algo_config['IFC'], self.algo_config['INOUTFC'],
                                                     self.algo_config['KHIN'], b='d', w=self.algo_config['DAT_WIDTH'])
        line3 = '{0} {1:{w}{b}} {2:{w}{b}}\n'.format(self.algo_config['HYDCHAR'], 0.0, 0.0, b='.1f',
                                                     w=self.algo_config['DAT_WIDTH'])
        lines.extend([line1, line2, line3])

        hourly_inflow = InflowAlgo.process_inflow(algo_input)
        i = 1.0
        for dt_index, rows in hourly_inflow.iterrows():
            value = float(rows['value'])
            line = '{0} {1:{w}{b}} {2:{w}{b}}\n'.format(self.algo_config['HYDCHAR'], i, value, b='.1f',
                                                        w=self.algo_config['DAT_WIDTH'])
            lines.append(line)
            i += 1.0

        with open(self.algo_config['init_wl_config']) as init_wl_conf_f:
            init_wls = init_wl_conf_f.readlines()
            for init_wl in init_wls:
                if len(init_wl.split()):
                    lines.append(init_wl)
        return lines

    @staticmethod
    def process_inflow(inflow_tms):
        if not isinstance(inflow_tms, pd.DataFrame):
            raise TypeError('Given timeseries is not a pandas data-frame of time, value columns')
        return inflow_tms.resample('H').max().dropna()


if __name__ == '__main__':
    outflow_config = Config('/home/nira/PycharmProjects/DI_Framework/flo2d_input_preparation/inflow/config.json')
    outflow_io = InflowIO(outflow_config)
    outflow_algo = InflowAlgo(outflow_io, outflow_config)
    outflow_algo.execute(start_dt=datetime(2018,1,1,0,0,0), base_dt=datetime(2018,1,3,0,0,0), end_dt=datetime(2018,1,5,8,1,0))