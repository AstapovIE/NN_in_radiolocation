import numpy as np

from simulation import RadarSystem
from simulation import Generator
from simulation import PBU, SimulationManager
from data_saver import Saver

# TODO !!! пока что закостылил init_position в generation


# ----------------------------------------------------- Initialization -----------------------------------------------
def gen_radars(air_env, n=16):
    # TODO optimize in future
    radars = []
    coords = [[5000, 5000, 0], [-5000, 5000, 0], [-5000, -5000, 0], [5000, -5000, 0],
              [10000, 10000, 0], [-10000, 10000, 0], [-10000, -10000, 0], [10000, -10000, 0],
              [10000, 5000, 0], [-10000, 5000, 0], [-10000, -5000, 0], [10000, -5000, 0],
              [5000, 10000, 0], [-5000, 10000, 0], [-5000, -10000, 0], [5000, -10000, 0]
              ]

    for coord in coords:
        radars.append(RadarSystem(position=np.array(coord), detection_period=detection_period,
                         detection_radius=detection_radius, air_env=air_env,
                        mean=np.array([0, np.random.uniform(1, 3), 0]),
                        error=np.array([np.random.uniform(1, 5), np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15)])
                                  )
                      )
    return radars


def generate_data(amount, t_start, t_end, detection_period):
    saver = Saver()

    for i in range(amount):
        air_env = gen.gen_traces()
        radars = gen_radars(air_env)
        sm = SimulationManager(air_env, PBU(radars))
        sm.run(t_start, t_end, detection_period)

        # сохраняем данные в папку /logs
        dataframes = sm.get_data()
        for m_j in range(len(dataframes)):
            saver.save_dataFrame(dataframes[m_j], f'traj{i+1}_radar{m_j + 1}')

t1 = 0
t2 = 10**6
detection_period = 1000

detection_radius = 400000

gen = Generator(detection_radius=detection_radius, start_time=t1, end_time=t2, num_samples=1, num_seg=2)

generate_data(3, t1,t2, detection_period)

