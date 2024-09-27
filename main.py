import numpy as np
import pandas as pd

from simulation import Time
from simulation import AirObject
from simulation import AirEnv
from simulation import RadarSystem
from simulation import Trajectory, TrajectorySegment
from simulation import PBU
from logger import Logger
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def calc_w(v: float, r: float):
    """
    Расчет угловой скорости рад / мс
    v - м / мс
    r - м
    """
    return v / r


def convert_velocity(V: float):
    """
    V in meters/s
    return: V in meters/ms
    """

    return V / 1000


# Создание объекта траектории
trajectory = Trajectory()

# Первый сегмент: прямолинейное движение с момента t=0 до t=100 по осям x, y, z
initial_position = [0, 0, 5]  # Начальная точка (x, y, z)

velocity = [convert_velocity(250),
            convert_velocity(220),
            convert_velocity(0)
            ]  # Скорости по x, y, z

trajectory.add_segment(TrajectorySegment(0, 400, initial_position, 'linear', velocity))

# Второй сегмент: движение по окружности
radius = 100
angular_velocity = calc_w(convert_velocity(300), radius)
vz = 0
trajectory.add_segment(TrajectorySegment(401, 1000, None, 'circular', [radius, angular_velocity, vz],
                                         previous_segment=trajectory.segments[-1]))

# # 3 сегмент: прямолинейное движение с момента t=701 до t=1000 по осям x, y, z
# trajectory.add_segment(TrajectorySegment(701, 1000, None, 'linear', velocity))

ao = AirObject(trajectory)
air_env = AirEnv([ao])
detection_radius = 200
radar1 = RadarSystem(position=np.array([50, 50, 0]), detection_radius=detection_radius, air_env=air_env)
radar2 = RadarSystem(position=np.array([-50, 50, 0]), detection_radius=detection_radius, air_env=air_env)
radar3 = RadarSystem(position=np.array([-50, -50, 0]), detection_radius=detection_radius, air_env=air_env)
radar4 = RadarSystem(position=np.array([50, -50, 0]), detection_radius=detection_radius, air_env=air_env)
pbu = PBU([radar1, radar2, radar3, radar4])

#----> решить что делать с системами координат, где ноль нужно ли считать относительно каждого радара. и тд..


t = Time()
t1, t2 = 0, 1000
for ms in tqdm(range(t1, t2)):
    ao.trigger()
    pbu.trigger()
    t.step()

logger = Logger()

dataframes = pbu.get_data()
for i in range(1, len(dataframes)+1):
    logger.log_dataFrame(dataframes[i-1], f'logs{i}')



# Визуализация
fig, ax = plt.subplots()
ax.set_xlim(-detection_radius*2 - 10, detection_radius*2 + 10)
ax.set_ylim(-detection_radius*2 - 10, detection_radius*2 + 10)

ax.add_patch(plt.Circle((-50, 50), detection_radius, color='b', fill=False, linestyle='--', label='Radar Range'))
ax.add_patch(plt.Circle((50, 50), detection_radius, color='orange', fill=False, linestyle='--', label='Radar Range'))
ax.add_patch(plt.Circle((50, -50), detection_radius, color='r', fill=False, linestyle='--', label='Radar Range'))
ax.add_patch(plt.Circle((-50, -50), detection_radius, color='g', fill=False, linestyle='--', label='Radar Range'))

xdata, ydata = [], []
line, = ax.plot(xdata, ydata, lw=2)

true_coord = ao.get_data()
for i in tqdm(range(len(true_coord))):
    xdata.append(true_coord['x_true'].iloc[i])
    ydata.append(true_coord['y_true'].iloc[i])

    line.set_data(xdata, ydata)

    # Перерисовываем график
    plt.draw()
    plt.pause(0.01)

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('AirObject Trajectory in XY Plane')
plt.show()
