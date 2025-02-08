import numpy as np
from .air_env import AirEnv
from .air_object import AirObject
from .trajectory import Trajectory, TrajectorySegment
from .unit import Unit
from .logger import Logger
from tools import Physic


class Generator(Unit):
    def __init__(
            self,
            detection_radius: float,
            start_time: float,
            end_time: float,
            neg_v_prob: float = 0.5,
            num_samples: int = 1,
            num_seg: int = 2,
            velocity_pool=np.arange(100, 201, 25),
            radius_pool=np.arange(5000, 10001, 500),
            logger=Logger(name='generation', log_file='log_file.txt'),
    ):
        super().__init__()
        self.__detection_radius = detection_radius
        self.__num_samples = num_samples
        self.__num_seg = num_seg
        self.velocity_pool = velocity_pool
        self.radius_pool = radius_pool
        self.neg_v_prob = neg_v_prob
        self.start_time = start_time
        self.end_time = end_time
        self.__logger = logger

    def trigger(self, **kwargs) -> None:
        pass

    import numpy as np

    def __generate_random_intervals(self, start_time, end_time, num_seg):
        total_duration = end_time - start_time
        random_points = np.sort(np.random.choice(range(1, total_duration), num_seg - 1, replace=False))

        time_intervals = [start_time] + list(random_points + start_time) + [end_time]
        return np.array(time_intervals)

    def __get_random_position(self, r, z_min=10 ** 3, z_max=1.2 * 10 ** 4) -> np.array:
        vec = np.random.normal(size=3)
        vec /= np.linalg.norm(vec)
        radius = np.random.uniform(0, 1) ** (1 / 3)
        vec = vec * r * radius * 0.2 # Множим вектор на радиус (КОСТЫЛЬ: и на конст 0.2, чтобы не вылетел за радар)
        vec[2] = np.clip(vec[2], z_min, z_max)
        return vec

    def __get_time_interval(self, time_intervals, num_seg) -> tuple:
        start_time = time_intervals[num_seg] + 1 if num_seg > 0 else time_intervals[num_seg]
        end_time = time_intervals[num_seg + 1]
        return start_time, end_time

    def __make_linear(self, trajectory, time_intervals, num_seg) -> TrajectorySegment:
        if num_seg == 0:
            sign = np.random.choice([-1, 1], p=[self.neg_v_prob, 1 - self.neg_v_prob])
            velocity = [
                sign * Physic.convert_velocity(np.random.choice(self.velocity_pool)),
                sign * Physic.convert_velocity(np.random.choice(self.velocity_pool)),
                0
            ]
        else:
            velocity = None
        start_time, end_time = self.__get_time_interval(time_intervals, num_seg)
        self.__logger.debug(f"Linear st_t = {start_time}, end_t = {end_time}")
        self.__logger.debug(f'V info num_seg = {num_seg}, v = {velocity}')
        if len(trajectory.get_segments()) != 0:
            return TrajectorySegment(start_time, end_time, None, 'linear', velocity,
                                     previous_segment=trajectory.get_segments()[-1])
        else:
            initial_position = self.__get_random_position(self.__detection_radius)
            # initial_position = np.array([0, 0, 5000])
            return TrajectorySegment(start_time, end_time, initial_position, 'linear', velocity)

    def __make_circular(self, trajectory, time_intervals, num_seg) -> TrajectorySegment:
        radius = np.random.choice(self.radius_pool)
        v = Physic.convert_velocity(np.random.choice(self.velocity_pool))
        angular_velocity = Physic.calc_w(v, radius)
        vz = 0
        start_time, end_time = self.__get_time_interval(time_intervals, num_seg)
        self.__logger.debug(f"Circular st_t = {start_time}, end_t = {end_time}")
        if len(trajectory.get_segments()) == 0:
            raise ValueError("Движение по окружности может быть только после прямолинейного")
        return TrajectorySegment(start_time, end_time, None, 'circular',
                                 [radius, angular_velocity, vz, np.random.choice([-1, 1])],
                                 previous_segment=trajectory.get_segments()[-1])

    def gen_traces(self) -> AirEnv:
        ae = AirEnv()
        for _ in range(self.__num_samples):
            trajectory = Trajectory()
            # self.__logger.debug(f"Id = {_}")
            time_intervals = self.__generate_random_intervals(self.start_time, self.end_time, self.__num_seg)
            for num_seg in range(self.__num_seg):
                motion_type = ['linear', 'circular'][num_seg % 2]
                if motion_type == 'linear':
                    trajectory.add_segment(self.__make_linear(trajectory, time_intervals, num_seg))
                else:
                    trajectory.add_segment(self.__make_circular(trajectory, time_intervals, num_seg))

            new_ao = AirObject(trajectory)
            ae.attach_air_object(new_ao)
        return ae

#
# class Generator(Unit):
#     def __init__(self, detection_radius: float, start_time: float, end_time: float, neg_v_prob: float = 0.1,
#                  num_samples: float = 1, num_seg: float = 2, velocity_pool=np.arange(200, 201, 25),
#                  radius_pool=np.arange(5000, 10001, 500)):
#         super().__init__()
#         self.__detection_radius = detection_radius
#         self.__num_samples = num_samples
#         self.__num_seg = num_seg
#         self.velocity_pool = velocity_pool
#         self.radius_pool = radius_pool
#         self.neg_v_prob = neg_v_prob
#         self.time_intervals = np.arange(start_time, end_time + 1, (end_time - start_time) / num_seg)
#
#     def trigger(self, **kwargs) -> None:
#         pass
#
#     def __get_random_position(self, r, z_min=10 ** 3, z_max=1.2 * 10 ** 4) -> np.array:
#         vec = np.random.normal(size=3)
#         vec /= np.linalg.norm(vec)  # Случайный единичный вектор
#
#         radius = np.random.uniform(0, 1) ** (1 / 3)  # Случайный радиус внутри объема единичной сферы
#
#         vec = vec * r * radius * 0.3 # Множим вектор на радиус (КОСТЫЛЬ: и на конст 0.3, чтобы не вылетел за радар)
#
#         # Ограничиваем третью координату z_max
#         vec[2] = np.clip(vec[2], z_min, z_max)
#
#         # return np.array([0, 0, 0])
#         return vec
#
#     def __get_time_interval(self, num_seg) -> tuple:
#         start_time = self.time_intervals[num_seg] + 1 if num_seg > 0 else self.time_intervals[num_seg]
#         end_time = self.time_intervals[num_seg + 1]
#         print(start_time, end_time, self.time_intervals)
#         return start_time, end_time
#
#     def __make_linear(self, trajectory, num_seg) -> TrajectorySegment:
#         sign = np.random.choice([-1, 1], p=[self.neg_v_prob, 1 - self.neg_v_prob])
#         velocity = [
#             sign * Physic.convert_velocity(np.random.choice(self.velocity_pool)),
#             sign * Physic.convert_velocity(np.random.choice(self.velocity_pool)),
#             0
#         ]  # Скорости по x, y, z
#
#         start_time, end_time = self.__get_time_interval(num_seg)
#         print(f"Linear st_t = {start_time}, end_t = {end_time}")
#         if len(trajectory.get_segments()):
#             return TrajectorySegment(start_time, end_time, None, 'linear', velocity)
#         else:
#             initial_position = self.__get_random_position(self.__detection_radius)
#             return TrajectorySegment(start_time, end_time, initial_position, 'linear', velocity)
#
#     def __make_circular(self, trajectory, num_seg) -> TrajectorySegment:
#         radius = np.random.choice(self.radius_pool)  # выбираем случайный радиус из допустимого набора
#         v = Physic.convert_velocity(
#             np.random.choice(self.velocity_pool))  # выбираем случайную скорость из допустимого набора
#         angular_velocity = Physic.calc_w(v, radius)  # угловая скорость
#         #vz = np.random.choice(np.arange(-10, 10, 2))  # моделируем скорость по оси z при движении по окружности
#         vz = 0
#         start_time, end_time = self.__get_time_interval(num_seg)
#         print(f"Circular st_t = {start_time}, end_t = {end_time}")
#         if len(trajectory.get_segments()) == 0:
#             raise ValueError("Движение по окружности может быть только после прямолинейного")
#         return TrajectorySegment(start_time, end_time, None, 'circular',
#                                  [radius, angular_velocity, vz, np.random.choice([-1, 1])],
#                                  previous_segment=trajectory.get_segments()[-1])
#
#     def gen_traces(self) -> AirEnv:
#         ae = AirEnv()
#         for _ in range(self.__num_samples):
#             trajectory = Trajectory()
#             print(f"Id = {_}")
#             for num_seg in range(self.__num_seg):
#                 motion_type = np.random.choice(['linear', 'circular'], [0, 1]) if num_seg >= 1 else 'linear'
#                 if motion_type == 'linear':
#                     trajectory.add_segment(self.__make_linear(trajectory, num_seg))
#                 else:
#                     trajectory.add_segment(self.__make_circular(trajectory, num_seg))
#             new_ao = AirObject(trajectory)
#             ae.attach_air_object(new_ao)
#         return ae
