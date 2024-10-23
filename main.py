import numpy as np
import pandas as pd

from simulation import AirObject
from simulation import AirEnv
from simulation import RadarSystem
from simulation import Trajectory, TrajectorySegment
from simulation import PBU, SimulationManager
from tools import Physic, MathStat
from logger import Logger

# пбу в 0, у каждого своя с-ма координаты относ и потом пересчитывать {{{ ЮСТИРОВКА(учет ошибок, чтобы дальше было лучше,
# исключ. систем. ошибки( нр неправ север и надо повернуть с-му) можно промоделировать это и тд)
#
# TODO а что если есть смещенность у какого-то из рлс, оценить её и учесть это (как-то по первым измерениям)
# 
# TODO насколько итоговая ошибка std (после оценки) лучше чем были
# TODO разные errors x2, x10

#  снова задуматься над физичностью полета в конкретных координатах
#  ещё раз подумать о характерных величинах в реал лайф


# Создание объекта траектории
trajectory = Trajectory()

# Первый сегмент: прямолинейное движение с момента t=0 до t=100 по осям x, y, z
initial_position = [0, 0, 5]  # Начальная точка (x, y, z)

velocity = [Physic.convert_velocity(200),
            Physic.convert_velocity(220),
            Physic.convert_velocity(0)]  # Скорости по x, y, z
trajectory.add_segment(TrajectorySegment(0, 300, initial_position, 'linear', velocity))

# Второй сегмент: движение по окружности
radius, vz = 100, 0
angular_velocity = Physic.calc_w(Physic.convert_velocity(300), radius)
trajectory.add_segment(TrajectorySegment(301, 1000, None, 'circular', [radius, angular_velocity, vz],
                                         previous_segment=trajectory.segments[-1]))

ao = AirObject(trajectory)
air_env = AirEnv([ao])
detection_radius = 200
e1 = 4
e2 = 4

# e3 = 1
radar1 = RadarSystem(position=np.array([100, 100, 0]), detection_radius=detection_radius, air_env=air_env, error=e1)
radar2 = RadarSystem(position=np.array([-100, 100, 0]), detection_radius=detection_radius, air_env=air_env, error=e2)
# radar3 = RadarSystem(position=np.array([50, 50, 0]), detection_radius=detection_radius, air_env=air_env, error=e3)

tm = 100

sm = SimulationManager(air_env, PBU([radar1, radar2]), ao)  # передавать {ao} временное решение
sm.run(0, tm)

logger = Logger()
# сохраняем данные в папку /logs
dataframes = sm.get_data()
for i in range(1, len(dataframes) + 1):
    logger.log_dataFrame(dataframes[i - 1], f'logs{i}')

df1 = pd.read_csv("logs/logs1.csv")
df2 = pd.read_csv("logs/logs2.csv")
# df3 = pd.read_csv("logs/logs3.csv")
list_of_df = [df1, df2]


# ----------------------------------------------------- MEAN -----------------------------------------------
e = np.zeros(tm)
for i in range(1, tm):
    x_true = list_of_df[0]["x_true"][i]
    x_avg = np.mean([df["x_measure"][i] for df in list_of_df])  # среднее координаты по всем радарам
    e[i] = round(abs(x_true - x_avg), 5)


# e_w = np.zeros(tm)
# # --- WEIGTHS
# v1 = 1 / (e1**2)
# v2 = 1 / (e2**2)
# ss = v1 + v2
# weigths = [v1 / ss, v2 / ss]
# for i in range(1, tm):
#     x_true = list_of_df[0]["x_true"][i]
#     x = [0] * 2
#     for q in range(2):
#         x[q] = list_of_df[q]["x_measure"][i] * weigths[q]
#     e_w[i] = round(abs(x_true - np.sum(x)), 5)



# ----------------------------------------------------- WEIGHTS -----------------------------------------------
e_w = np.zeros(tm)
sigmas = sm.get_radar_errors()
for i in range(1, tm):
    x_true = list_of_df[0]["x_true"][i]
    x_estimated = MathStat.weighted_estimator([df["x_measure"][i] for df in list_of_df], sigmas)
    e_w[i] = round(abs(x_true - np.sum(x_estimated)), 5)


# --- экспоненциальное сглаживание
# eps = np.array([0.7, 0.6])
# m = 2
# dx_volna = np.ndarray((tm, m))
# dx_volna[0] = [0, 0]
#
# e_eps = np.zeros(tm)
# for i in range(1, tm):
#     x_true = list_of_df[0]["x_true"][i]
#     x = [df["x_measure"][i] for df in list_of_df]
#     x_avg = np.mean(x)  # среднее координаты по всем радарам
#     dx = np.array([(x_avg - df["x_measure"][i]) for df in list_of_df])
#     tmp = []
#     for q in range(m):
#         tmp.append((1 - eps[q]) * dx[q] + eps[q] * dx_volna[i][q])
#     dx_volna[i] = tmp
#
#     x_volna = np.mean(x + dx_volna)
#     e_eps[i] = round(abs(x_true - x_volna), 8)
# print(dx_volna)


# ----------------------------------------------------- VIZUAL -----------------------------------------------

import matplotlib.pyplot as plt

plt.plot(np.arange(5, tm), e[5:], color="r", label='mean')
plt.plot(np.arange(1, tm), e_w[1:], color="b", alpha=0.5, label='weights')
# plt.plot(np.arange(1, tm), e_w2[1:], color="g", alpha=0.5, label='weights2')
plt.legend()
plt.grid()
plt.show()
