import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, NoReturn
import matplotlib.pyplot as plt

from .time import Time
from simulation import AirEnv, PBU


class SimulationManager:

    def __init__(self, air_env: AirEnv, pbu: PBU, first_air_object=None) -> NoReturn:
        self.__air_env = air_env
        self.__pbu = pbu
        # self.ao = first_air_object  # временное решение

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
        t.set_dt(dt)

        if progress:
            progressbar = tqdm(range(t_min, t_max + 1, dt))
            progressbar.set_description('Running system')
        else:
            progressbar = range(t_min, t_max + 1, dt)
        for _ in progressbar:
            self.__pbu.trigger()
            # self.__air_env.trigger()
            t.step()
        t.reset()

    def get_data(self) -> List[pd.DataFrame]:
        """
        Получение данных ПУ
        """
        return self.__pbu.get_data()

    def get_radars_errors(self):
        return self.__pbu.get_errors()

    # def set_radars_errors(self, new_errors: List[np.array]):
    #     return self.__pbu.set_errors(new_errors)

    def get_radars_positions(self):
        return self.__pbu.get_radars_position()

    def get_radars_detection_radius(self):
        return self.__pbu.get_radars_detection_radius()

    def visualize(self):
        data = self.get_data()[0] # берем истинные координаты из любого (тут из первого) радара
        radii = self.get_radars_detection_radius() # радиусы радаров
        positions = self.get_radars_positions() # координаты радаров

        # Визуализация
        fig, ax = plt.subplots()

        for i in range(self.__pbu.get_num_radars()):
            if i == 0:
                ax.add_patch(plt.Circle(positions[i][:-1], radii[i], fill=False, linestyle='--', label='Зона обнаружения радара'))
            else:
                ax.add_patch(plt.Circle(positions[i][:-1], radii[i], fill=False, linestyle='--'))

        plt.plot(data['x_true'], data['y_true'], label=f"Траектория ВО")

        # plt.draw()
        plt.xlabel('X Координата (м)')
        plt.ylabel('Y Координата (м)')
        plt.title('Траектория ВО в плоскости Х У')
        plt.tight_layout()
        plt.legend()
        plt.show()