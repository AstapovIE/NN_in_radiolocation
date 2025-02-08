import numpy as np


class Physic:
    @staticmethod
    def calc_w(v: float, r: float):
        """
        Расчет угловой скорости рад / мс
        v - м / мс
        r - м
        """
        return v / r

    @staticmethod
    def convert_velocity(v: float):
        """
        V in meters/s
        return: V in meters/ms
        """
        return v / 1000

    @staticmethod
    def to_radians(angle) -> float:
        """
        Перевод угла в градусах в радианы
        """
        return angle / 180 * np.pi

    @staticmethod
    def to_sphere_coord(x, y, z) -> tuple:
        """
        Перевод декартовой системы координат в сферическую
        :return: tuple = (r, theta, fi)
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) # угол наклона относительно оси z [0, pi]
        fi = np.atan2(y, x) # угол в плоскости x, y [-pi, pi)
        return (r, theta, fi)

    @staticmethod
    def to_cartesian_coord(r, theta, fi) -> tuple:
        """
        Перевод сферической системы координат в декартову
        :return: tuple = (x, y, z)
        """
        x = r * np.sin(theta) * np.cos(fi)
        y = r * np.sin(theta) * np.sin(fi)
        z = r * np.cos(theta)
        return (x, y, z)

    @staticmethod
    def normalize_theta(theta):
        """Нормализация угла theta в пределах от 0 до pi."""
        return np.clip(theta, 0, np.pi)

    @staticmethod
    def normalize_fi(fi):
        """Нормализация угла fi в пределах от -pi до pi."""
        return (fi + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def cartesian_to_spherical_velocity(v_x, v_y, v_z, x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        fi = np.arctan2(y, x)

        v_r = (v_x * x + v_y * y + v_z * z) / r
        v_theta = (-v_x * np.sin(fi) + v_y * np.cos(fi) + v_z * np.sin(theta)) / r
        v_fi = (v_x * np.sin(fi) + v_y * np.cos(fi)) / (r * np.sin(theta))

        return v_r, v_theta, v_fi

    @staticmethod
    def spherical_to_cartesian_velocity(v_r, v_theta, v_fi, r, theta, fi):
        v_x = v_r * np.sin(theta) * np.cos(fi) + r * v_theta * np.cos(theta) * np.cos(fi) - r * v_fi * np.sin(
            theta) * np.sin(fi)
        v_y = v_r * np.sin(theta) * np.sin(fi) + r * v_theta * np.cos(theta) * np.sin(fi) + r * v_fi * np.sin(
            theta) * np.cos(fi)
        v_z = v_r * np.cos(theta) - r * v_theta * np.sin(theta)

        return v_x, v_y, v_z

