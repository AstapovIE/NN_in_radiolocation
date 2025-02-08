import numpy as np
import pandas as pd

from .unit import Unit
from .air_env import AirEnv
from .logger import Logger
from tools import Physic


class RadarSystem(Unit):

    def __init__(self,
                 logger = Logger(name='radar_system', log_file='log_file.txt'),
                 position: np.array = np.array([0, 0, 0]),
                 detection_radius: float = 10000,
                 mean: np.array=np.array([0., 0., 0.]),
                 error: np.array=np.array([0., 0., 0.]),
                 air_env: AirEnv = None,
                 detection_fault_probability: float = 0.0,
                 detection_period: int = 1,
                 detection_delay: int = 0) -> None:
        """
                position: позиция радара
                detection_radius: радиус обнаружения в метрах
                mean: вектор смещений локатора по сферическим координатам (r_error (м), theta_error (градусы), fi_error (градусы))
                error: вектор ошибок локатора по сферическим координатам (r_error (м), theta_error (градусы), fi_error (градусы))
                air_env: объект воздушной обстановки
                detection_fault_probability: вероятность ошибки обнаружения
                detection_period: частота обращения локатора к цели (мс)
                detection_delay: задержка обрашения (мс)
        """

        super().__init__()
        self.__logger = logger
        self.__position = np.array(position, dtype=float)
        self.__detection_fault_probability = detection_fault_probability
        self.__detection_period = detection_period
        self.__detection_delay = detection_delay % detection_period
        self.__detection_radius = detection_radius

        self.__r_mean, self.__theta_mean, self.__fi_mean = mean
        self.__theta_mean = Physic.to_radians(self.__theta_mean)
        self.__fi_mean = Physic.to_radians(self.__fi_mean)

        self.__r_error, self.__theta_error, self.__fi_error = error
        self.__theta_error = Physic.to_radians(self.__theta_error)
        self.__fi_error = Physic.to_radians(self.__fi_error)


        self.__air_env = air_env

        self.__data_dtypes = {
            'id': 'int64',
            'time': 'int64',

            'x_true': 'float64',
            'y_true': 'float64',
            'z_true': 'float64',
            'x_measure': 'float64',
            'y_measure': 'float64',
            'z_measure': 'float64',

            'r_true': 'float64',
            'theta_true': 'float64',
            'fi_true': 'float64',
            'r_measure': 'float64',
            'theta_measure': 'float64',
            'fi_measure': 'float64',

             'v_x_true': 'float64',
            'v_y_true': 'float64',
            'v_z_true': 'float64',
            'v_x_measure': 'float64',
            'v_y_measure': 'float64',
            'v_z_measure': 'float64',

            'v_r_true': 'float64',
            'v_fi_true': 'float64',
            'v_theta_true': 'float64',
            'v_r_measure': 'float64',
            'v_fi_measure': 'float64',
            'v_theta_measure': 'float64',

            # 'x_err': 'float64',
            # 'y_err': 'float64',
            # 'z_err': 'float64',
            'r_mean': 'float64',
            'fi_mean': 'float64',
            'theta_mean': 'float64',
            'r_error': 'float64',
            'theta_error': 'float64',
            'fi_error': 'float64',

        }
        self.__data = pd.DataFrame(columns=list(self.__data_dtypes.keys())).astype(self.__data_dtypes)

    def trigger(self) -> None:
        if self.time.get_time() % self.__detection_period == self.__detection_delay:
            if np.random.choice([False, True],
                                p=[self.__detection_fault_probability, 1.0 - self.__detection_fault_probability]):
                self.detect_air_objects()

    def detect_air_objects(self) -> None:
        prev_detect = None
        if len(self.__data) != 0:
            air_objects_count = self.__air_env.get_air_objects_count()
            prev_detect = self.__data.tail(air_objects_count)
            # print(f'prev_detect = {prev_detect}')
            prev_detect = prev_detect.set_index(prev_detect['id'])
            # print(f'prev_detect_set = {prev_detect}')

        # Получение положений всех ВО в наблюдаемой AirEnv
        detections = self.__air_env.air_objects_dataframe()

        # Фильтрация ВО с координатами вне области наблюдения
        p = self.__position
        r = self.__detection_radius
        detections['is_observed'] = detections.apply(lambda row: np.sqrt(
            (row['x_true'] - p[0]) ** 2 + (row['y_true'] - p[1]) ** 2 + (row['z_true'] - p[2]) ** 2) <= r,axis=1
        )
        detections = detections[detections['is_observed']]
        detections.drop(columns=['is_observed'], inplace=True)

        detections['time'] = self.time.get_time()
        detections['r_true'], detections['theta_true'], detections['fi_true'] = Physic.to_sphere_coord(
            detections['x_true'], detections['y_true'], detections['z_true'])




        detections['r_measure'] = detections['r_true'] + np.random.normal(self.__r_mean, self.__r_error, len(detections))
        detections['theta_measure'] = Physic.normalize_theta(detections['theta_true'] + np.random.normal(self.__theta_mean, self.__theta_error, len(detections)))
        detections['fi_measure'] = Physic.normalize_fi(detections['fi_true'] + np.random.normal(self.__fi_mean, self.__fi_error, len(detections)))

        detections['x_measure'], detections['y_measure'], detections['z_measure'] = Physic.to_cartesian_coord(
            detections['r_measure'], detections['theta_measure'], detections['fi_measure'])

        detections['r_error'] = self.__r_error
        detections['theta_error'] = self.__theta_error
        detections['fi_error'] = self.__fi_error

        detections['r_mean'] = self.__r_mean
        detections['theta_mean'] = self.__theta_mean
        detections['fi_mean'] = self.__fi_mean



        # detections['x_measure'] = detections['x_true'] + np.random.normal(self.__mean, self.__error, len(detections))
        # detections['y_measure'] = detections['y_true'] + np.random.normal(self.__mean, self.__error, len(detections))
        # detections['z_measure'] = detections['z_true'] + np.random.normal(self.__mean, self.__error, len(detections))
        # detections['r_measure'], detections['fi_measure'], detections['psi_measure'] = Physic.to_sphere_coord(
        #     detections['x_measure'], detections['y_measure'], detections['z_measure'])
        #
        # detections['x_err'] = self.__error
        # detections['y_err'] = self.__error
        # detections['z_err'] = self.__error


        # Вычисление скоростей
        for coord in (
                'x_true',
                'y_true',
                'z_true',
        ):
            if prev_detect is None:
                detections[f'v_{coord}'] = None
            else:
                dt = (detections['time'] - prev_detect['time'])  # шаг по времени в секундах
                detections[f'v_{coord}'] = (detections[coord] - prev_detect[coord]) / dt

        if prev_detect is None:
            detections[f'v_r_true'] = None
            detections[f'v_theta_true'] = None
            detections[f'v_fi_true'] = None
        else:
            detections[f'v_r_true'], detections[f'v_theta_true'], detections[
                f'v_fi_true'] = Physic.cartesian_to_spherical_velocity(detections[f'v_x_true'], detections[f'v_y_true'],
                                                                       detections[f'v_z_true'], detections['x_true'],
                                                                       detections[f'y_true'], detections[f'z_true'])

        for coord in (
                'x_measure',
                'y_measure',
                'z_measure',
        ):
            if prev_detect is None:
                detections[f'v_{coord}'] = None
            else:
                dt = detections['time'] - prev_detect['time']  # шаг по времени в секундах
                detections[f'v_{coord}'] = (detections[coord] - prev_detect[
                    coord]) / dt  # Вычисление скорости на текущем цикле обзора

        if prev_detect is None:
            detections[f'v_r_measure'] = None
            detections[f'v_theta_measure'] = None
            detections[f'v_fi_measure'] = None
        else:
            detections[f'v_r_measure'], detections[f'v_theta_measure'], detections[
                f'v_fi_measure'] = Physic.cartesian_to_spherical_velocity(detections[f'v_x_measure'],
                                                                          detections[f'v_y_measure'],
                                                                          detections[f'v_z_measure'],
                                                                          detections[f'x_measure'],
                                                                          detections[f'y_measure'],
                                                                          detections[f'z_measure'])






        # Вычисление скоростей
        # for coord in (
        #         'x_true',
        #         'y_true',
        #         'z_true',
        #         'x_measure',
        #         'y_measure',
        #         'z_measure',
        #         'r_true',
        #         'fi_true',
        #         'theta_true',
        #         'r_measure',
        #         'fi_measure',
        #         'theta_measure'
        # ):
        #     detections[f'v_{coord}_extr'] = None if len(self.__data) == 0 else (detections[coord] -
        #                                                                         self.__data.iloc[len(self.__data) - 1][
        #                                                                             coord]) / self.time.get_dt()

        # Concat new detections with data
        self.__concat_data(detections)

    def __concat_data(self, df: pd.DataFrame) -> None:
        df = df[list(self.__data_dtypes.keys())].astype(self.__data_dtypes)
        if len(self.__data) == 0:
            self.__data = df
        else:
            self.__data = pd.concat([self.__data, df])
            self.__data.reset_index(inplace=True, drop=True)

    def get_data(self) -> pd.DataFrame:
        return self.__data.copy()

    def get_position(self):
        return self.__position

    def get_detection_radius(self):
        return self.__detection_radius

    def get_mean(self):
        return (self.__r_mean, self.__theta_mean, self.__fi_mean) #self.__mean

    def get_error(self):
        return (self.__r_error, self.__theta_error, self.__fi_error) #self.__error

    def set_error(self, new_error):
        self.__error = new_error if new_error>0 else (new_error + 10**-10)  # чтобы избежать деления на 0

    def clear_data(self) -> None:
        self.__data = self.__data.iloc[0:0]

    def set_air_environment(self, air_env: AirEnv) -> None:
        self.__air_env = air_env

    def set_detection_fault_probability(self, detection_fault_probability: float) -> None:
        self.__detection_fault_probability = detection_fault_probability

    def set_detection_period(self, detection_period: int) -> None:
        self.__detection_period = detection_period

    def repr(self) -> str:
        return '<RadarSystem: position={}, detection_radius={}, error={}>'.format(
            self.__position, self.__detection_radius, self.__error
        )
