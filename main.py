import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation import RadarSystem
from simulation import Generator
from simulation import PBU, SimulationManager
from tools import MathStat, DynamicAlignment
from data_saver import Saver

#  снова задуматься над физичностью полета в конкретных координатах
#  ещё раз подумать о характерных величинах в реал лайф


# какую ошибку задавать? в декарт или сфер? (расчеты графиков были для декарт)

# ----------------------------------------------------- Initialization -----------------------------------------------
t1 = 0
t2 = 10**6
detection_period = 1000

num_points = t2//detection_period

detection_radius = 400000
gen = Generator(detection_radius=detection_radius, start_time=t1, end_time=t2, num_samples=1, num_seg=3)
air_env = gen.gen_traces()

e1 = 2
e2 = 7
e3 = 25
e4 = 23

bog_e1 = [e1, e1, e1]
bog_e2 = [e2, e2, e2]
bog_e3 = [e3, e3, e3]
bog_e4 = [e4, e4, e4]

loc1 = 12
loc2 = 7
loc3 = -5
loc4 = 2

n_radars = 4

# error=np.array([1-5, 0.05-0.15, 0.1])
# mean=np.array([0-1, 1-3, 1-3])
# пока только по одному углу


radar1 = RadarSystem(position=np.array([10000, 10000, 0]), detection_radius=detection_radius, detection_period=detection_period, air_env=air_env,
                     mean=np.array([12., 0.2, 0.1]), error=np.array([1., 0.1, 0.1]))
radar2 = RadarSystem(position=np.array([-10000, 10000, 0]), detection_radius=detection_radius, detection_period=detection_period, air_env=air_env,
                     mean=np.array([7., 0.3, 0.3]), error=np.array([1., 0.1, 0.1]))
radar3 = RadarSystem(position=np.array([-10000, -10000, 0]), detection_radius=detection_radius, detection_period=detection_period, air_env=air_env,
                     mean=np.array([5., 0.2, 0.3]), error=np.array([1., 0.1, 0.1]))
radar4 = RadarSystem(position=np.array([10000, -10000, 0]), detection_radius=detection_radius, detection_period=detection_period, air_env=air_env,
                     mean=np.array([2., 0.5, 0.1]), error=np.array([1., 0.1, 0.1]))

# radar1 = RadarSystem(position=np.array([10000, 10000, 0]), detection_radius=detection_radius, detection_period=detection_period, air_env=air_env,
#                      mean=loc1, error=e1)
# radar2 = RadarSystem(position=np.array([-10000, 10000, 0]), detection_radius=detection_radius, detection_period=detection_period, air_env=air_env,
#                      mean=loc2, error=e2)
# radar3 = RadarSystem(position=np.array([-10000, -10000, 0]), detection_radius=detection_radius, detection_period=detection_period, air_env=air_env,
#                      mean=loc3, error=e3)
# radar4 = RadarSystem(position=np.array([10000, -10000, 0]), detection_radius=detection_radius, detection_period=detection_period, air_env=air_env,
#                      mean=loc4, error=e4)

sm = SimulationManager(air_env, PBU([radar1, radar2, radar3, radar4]))  # передавать {ao} временное решение
sm.run(t1, t2, detection_period)

# визуализируем радары
sm.visualize()

saver = Saver()
# сохраняем данные в папку /logs
dataframes = sm.get_data()
for i in range(len(dataframes)):
    saver.save_dataFrame(dataframes[i], f'logs{i + 1}')

# Считываем данные
# list_of_df = [pd.read_csv(f"logs/logs{i + 1}.csv") for i in range(n_radars)]

# ----------------------------------------------------- DynamicAlignment -----------------------------------------------


# x_true = np.zeros(num_points)
# x_estimated_w_align = np.zeros(num_points)
# x_estimated = np.zeros(num_points)
# x_mean = np.zeros(num_points)
#
# sigmas = sm.get_radars_errors()
# print("sigmas =      ", sigmas)
# print("result_sigma after weighted_estimator =", MathStat.find_res_sigma(sigmas))
# smoother = DynamicAlignment(n_radars)
# smoother.update_ksi(sigmas)
#
# e_estimated_w_align = np.zeros(num_points)
# e_estimated = np.zeros(num_points)
# e_mean = np.zeros(num_points)
#
# X = np.zeros((n_radars, num_points))
# print(num_points)
# for i in range(1, num_points):
#
#     # берем истинные координаты
#     x_true[i] = list_of_df[0]["x_true"][i]
#
#     # измеренные координаты от всех радаров
#     coords = [df["x_measure"][i] for df in list_of_df]
#     X[:, i] = np.array(coords)
#     # сгладим их
#     aligned_coords = smoother.compute_alignments(coords)
#
#     # 1) просто среднее
#     x_mean[i] = np.mean(coords)
#     e_mean[i] = round(abs(x_true[i] - x_mean[i]), 5)
#
#     # 2) взвешенная оценка без сглаживания
#     x_estimated[i] = MathStat.weighted_estimator(coords, sigmas)
#     e_estimated[i] = round(abs(x_true[i] - x_estimated[i]), 5)
#
#     # 3) взвешенная оценка + сглаживание
#     x_estimated_w_align[i] = MathStat.weighted_estimator(aligned_coords, sigmas)
#     e_estimated_w_align[i] = round(abs(x_true[i] - x_estimated_w_align[i]), 5)
#
# step = 10
# plt.plot(np.arange(0, t2, step), x_true[::step], label='Истинные координаты')
# plt.plot(np.arange(0, t2, step), x_mean[::step], label='Среднее')
# plt.plot(np.arange(0, t2, step), x_estimated[::step], label='Взвешенная оценка')
# plt.plot(np.arange(0, t2, step), x_estimated_w_align[::step], label='Взвешенная оценка сглаженных координат')
# plt.title("Зависимость координаты 'x' по времени")
# plt.ylabel("Координата 'x' (м)")
# plt.xlabel("Время (м/c)")
# plt.legend()
# plt.grid()
# plt.show()

# start = 0
# print(len(np.arange(start, num_points, step)))
# print(np.arange(start, num_points, step))
# print(e_mean)

# plt.plot(np.arange(start, num_points, step), e_mean[start::step], label='Среднее', alpha=0.8)
# plt.plot(np.arange(start, num_points, step), e_estimated[start::step], label='Взвешенная оценка')
# plt.plot(np.arange(start, num_points, step), e_estimated_w_align[start::step], color='r', label='Взвешенная оценка сглаженных координат')
# plt.legend()
# plt.title("Модуль ошибки оценки истинной координаты")
# plt.ylabel("Ошибка (м)")
# plt.xlabel("Время (c)")
# plt.grid()
# plt.show()