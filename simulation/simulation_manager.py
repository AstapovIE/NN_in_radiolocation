import pandas as pd
from tqdm import tqdm
from typing import List, NoReturn
import matplotlib.pyplot as plt

from .time import Time
from simulation import AirEnv, PBU


def vizual(data, detection_radius):
    # Визуализация
    fig, ax = plt.subplots()
    ax.set_xlim(-detection_radius * 2 - 10, detection_radius * 2 + 10)
    ax.set_ylim(-detection_radius - 10, detection_radius * 2 + 10)

    ax.add_patch(plt.Circle((-50, 50), detection_radius, color='b', fill=False, linestyle='--', label='Radar Range'))
    ax.add_patch(plt.Circle((50, 50), detection_radius, color='r', fill=False, linestyle='--', label='Radar Range'))

    xdata, ydata = [], []
    line, = ax.plot(xdata, ydata, lw=2)

    for i in tqdm(range(len(data))):
        xdata.append(data['x_true'].iloc[i])
        ydata.append(data['y_true'].iloc[i])

        line.set_data(xdata, ydata)

        # Перерисовываем график
        plt.draw()
        plt.pause(0.01)

    plt.xlabel('X Coordinate meters')
    plt.ylabel('Y Coordinate meters')
    plt.title('AirObject Trajectory in XY Plane')
    plt.show()


class SimulationManager:

    def __init__(self, air_env: AirEnv, pbu: PBU, first_air_object) -> NoReturn:
        self.__air_env = air_env
        self.__pbu = pbu
        self.ao = first_air_object  # временное решение

    def run(self, t_min: int, t_max: int, dt: int = 1, progress: bool = True) -> NoReturn:
        """
        Запуск моделирования в заданном интервале времени с заданным шагом
        :param t_min: начальный момент времени (в мс)
        :param t_max: конечный момент времени (в мс)
        :param dt: шаг по времени (в мс)
        :param progress: показывать прогресс процесса моделирования (если True) или нет (если False)
        """
        t = Time()
        t.set(t_min)

        if progress:
            progressbar = tqdm(range(t_min, t_max + 1, dt))
            progressbar.set_description('Running system')
        else:
            progressbar = range(t_min, t_max + 1, dt)
        for _ in progressbar:
            self.__pbu.trigger()
            self.__air_env.trigger()
            t.step()

        detect_radius = 200  # временное решение
        vizual(self.ao.get_data(), detect_radius)  # временное решение

    def get_data(self) -> List[pd.DataFrame]:
        """
        Получение данных ПУ
        """
        return self.__pbu.get_data()

    def get_radar_errors(self):
        return self.__pbu.get_errors()
