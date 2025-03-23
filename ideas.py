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

# ----------- Создание объекта траектории
# trajectory = Trajectory()
#
# # Первый сегмент: прямолинейное движение с момента t=0 до t=100 по осям x, y, z
# initial_position = [0, 0, 5]  # Начальная точка (x, y, z)
#
# velocity = [Physic.convert_velocity(220),
#             Physic.convert_velocity(220),
#             Physic.convert_velocity(0)]  # Скорости по x, y, z
# trajectory.add_segment(TrajectorySegment(0, 300, initial_position, 'linear', velocity))
#
# # Второй сегмент: движение по окружности
# radius, vz = 100, 0
# angular_velocity = Physic.calc_w(Physic.convert_velocity(300), radius)
# trajectory.add_segment(TrajectorySegment(301, 1000, None, 'circular', [radius, angular_velocity, vz],
#                                          previous_segment=trajectory.segments[-1]))


# ----------------------------------------------------- coords_vizual -----------------------------------------------
# X = np.zeros((n_radars, t2))
# x_true = np.zeros(t2)
# x_estimated = np.zeros(t2)
# sigmas = sm.get_radar_errors()
# print("sigmas =      ", sigmas)
# print("result_sigma =", MathStat.find_res_sigma(sigmas))
#
# e_w = np.zeros(t2)
#
# popravka = np.zeros(n_radars)
# print("popravka each 20 steps")
# for i in range(1, t2):
#     if (i+1)%20 == 0:
#         delta = np.mean(X[0, (i//2 - 1):i]) - np.mean( [np.mean(X[1, (i//2 - 1):i]), np.mean(X[2, (i//2 - 1):i]), np.mean(X[3, (i//2 - 1):i])] )
#         # delta = np.mean(X[0, (i//2 - 1):i]) - np.mean( [np.mean(X[1, (i//2 - 1):i])])#, np.mean(X[2, (i//2 - 1):i]), np.mean(X[3, (i//2 - 1):i])] )
#         popravka[0] = delta
#         print(popravka)
#
#     x_true[i] = list_of_df[0]["x_true"][i]
#     X[:, i] = [df["x_measure"][i] for df in list_of_df]
#     x_estimated[i] = MathStat.weighted_estimator([df["x_measure"][i] for df in list_of_df] - popravka, sigmas)
#     e_w[i] = round(abs(x_true[i] - x_estimated[i]), 5)
#
# x1 = X[0, :]
# x2 = X[1, :]
# x3 = X[2, :]
# x4 = X[3, :]
#
# print(np.mean(x1), np.mean( [np.mean(x2), np.mean(x3), np.mean(x4)] ))
#
#
#
# for i in range(n_radars):
#     plt.plot(np.arange(t2), X[i, :], label='meas')

# import numpy as np
#
# print(np.random.uniform(0.05, 0.15))



import numpy as np

# Фиксируем параметры цели
r_mean = 5000  # метров
theta_base = np.radians(45)  # Базовый азимут цели (рад)
phi_base = np.radians(30)  # Базовый угол возвышения (рад)

# Списки для хранения ошибок
dx_list, dy_list, dz_list = [], [], []

# Количество экспериментов
num_experiments = 100

for _ in range(num_experiments):
    # Генерация случайных ошибок радара
    theta_mean = np.radians(np.random.uniform(1, 3))  # Смещение азимута (рад)
    phi_mean = np.radians(0)  # Смещение по возвышению (рад)
    r_error = np.random.uniform(1, 5)  # Ошибка по радиусу (м)
    theta_error = np.radians(np.random.uniform(0.05, 0.15))  # Ошибка по азимуту (рад)
    phi_error = np.radians(np.random.uniform(0.05, 0.15))  # Ошибка по возвышению (рад)

    # Углы цели с учетом смещения
    theta = theta_base + theta_mean
    phi = phi_base + phi_mean

    # Ошибки в декартовых координатах
    dx = abs(np.sin(theta) * np.cos(phi) * r_error + r_mean * np.cos(theta) * np.cos(phi) * theta_error - r_mean * np.sin(theta) * np.sin(phi) * phi_error)
    dy = abs(np.sin(theta) * np.sin(phi) * r_error + r_mean * np.cos(theta) * np.sin(phi) * theta_error + r_mean * np.sin(theta) * np.cos(phi) * phi_error)
    dz = abs(np.cos(theta) * r_error - r_mean * np.sin(phi) * theta_error)

    # dx = abs(
    #     np.cos(theta) * np.cos(phi) * r_error - r_mean * np.sin(theta) * np.cos(phi) * theta_error - r_mean * np.cos(
    #         theta) * np.sin(phi) * phi_error)
    # dy = abs(
    #     np.sin(theta) * np.cos(phi) * r_error + r_mean * np.cos(theta) * np.cos(phi) * theta_error - r_mean * np.sin(
    #         theta) * np.sin(phi) * phi_error)
    # dz = abs(np.sin(phi) * r_error + r_mean * np.cos(phi) * phi_error)

    # Сохраняем результаты
    dx_list.append(dx)
    dy_list.append(dy)
    dz_list.append(dz)

# Вычисляем статистику
dx_min, dx_max, dx_mean = round(min(dx_list), 2), round(max(dx_list), 2), round(np.mean(dx_list), 2)
dy_min, dy_max, dy_mean = round(min(dy_list), 2), round(max(dy_list), 2), round(np.mean(dy_list), 2)
dz_min, dz_max, dz_mean = round(min(dz_list), 2), round(max(dz_list), 2), round(np.mean(dz_list), 2)

print("     min              max                     avg  ")
print((dx_min, dx_max, dx_mean))
print((dy_min, dy_max, dy_mean))
print((dz_min, dz_max, dz_mean))

# import numpy as np
#
# # Исходные данные
# r = 10000  # м
# theta = np.deg2rad(45)  # рад
# phi = np.deg2rad(45)  # рад
#
# # Ошибки в сферических координатах
# delta_r = 0  # м
# sigma_r = 3  # м
#
# delta_theta = np.deg2rad(2)  # рад
# sigma_theta = np.deg2rad(0.1)  # рад
#
# delta_phi = 0  # рад
# sigma_phi = np.deg2rad(0.1)  # рад
#
# # Частные производные
# dx_dr = np.sin(theta) * np.cos(phi)
# dx_dtheta = r * np.cos(theta) * np.cos(phi)
# dx_dphi = -r * np.sin(theta) * np.sin(phi)
#
# dy_dr = np.sin(theta) * np.sin(phi)
# dy_dtheta = r * np.cos(theta) * np.sin(phi)
# dy_dphi = r * np.sin(theta) * np.cos(phi)
#
# dz_dr = np.cos(theta)
# dz_dtheta = -r * np.sin(theta)
#
# # Смещение в декартовых координатах
# delta_x = dx_dr * delta_r + dx_dtheta * delta_theta + dx_dphi * delta_phi
# delta_y = dy_dr * delta_r + dy_dtheta * delta_theta + dy_dphi * delta_phi
# delta_z = dz_dr * delta_r + dz_dtheta * delta_theta
#
# # СКО в декартовых координатах
# sigma_x = np.sqrt((dx_dr * sigma_r)**2 + (dx_dtheta * sigma_theta)**2 + (dx_dphi * sigma_phi)**2)
# sigma_y = np.sqrt((dy_dr * sigma_r)**2 + (dy_dtheta * sigma_theta)**2 + (dy_dphi * sigma_phi)**2)
# sigma_z = np.sqrt((dz_dr * sigma_r)**2 + (dz_dtheta * sigma_theta)**2)
#
# # Вывод результатов
# print(f"Смещение в декартовых координатах:")
# print(f"Δx = {delta_x:.2f} м, Δy = {delta_y:.2f} м, Δz = {delta_z:.2f} м")
#
# print(f"СКО в декартовых координатах:")
# print(f"σx = {sigma_x:.2f} м, σy = {sigma_y:.2f} м, σz = {sigma_z:.2f} м")


# Функция для перевода ошибок из сферических в декартовы координаты
def spherical_to_cartesian_errors(r, theta, phi, delta_r, sigma_r, delta_theta, sigma_theta, delta_phi, sigma_phi):
    # Частные производные
    dx_dr = np.sin(theta) * np.cos(phi)
    dx_dtheta = r * np.cos(theta) * np.cos(phi)
    dx_dphi = -r * np.sin(theta) * np.sin(phi)

    dy_dr = np.sin(theta) * np.sin(phi)
    dy_dtheta = r * np.cos(theta) * np.sin(phi)
    dy_dphi = r * np.sin(theta) * np.cos(phi)

    dz_dr = np.cos(theta)
    dz_dtheta = -r * np.sin(theta)

    # Смещение в декартовых координатах
    delta_x = dx_dr * delta_r + dx_dtheta * delta_theta + dx_dphi * delta_phi
    delta_y = dy_dr * delta_r + dy_dtheta * delta_theta + dy_dphi * delta_phi
    delta_z = dz_dr * delta_r + dz_dtheta * delta_theta

    # СКО в декартовых координатах
    sigma_x = np.sqrt((dx_dr * sigma_r)**2 + (dx_dtheta * sigma_theta)**2 + (dx_dphi * sigma_phi)**2)
    sigma_y = np.sqrt((dy_dr * sigma_r)**2 + (dy_dtheta * sigma_theta)**2 + (dy_dphi * sigma_phi)**2)
    sigma_z = np.sqrt((dz_dr * sigma_r)**2 + (dz_dtheta * sigma_theta)**2)

    return delta_x, delta_y, delta_z, sigma_x, sigma_y, sigma_z

# Параметры
r = 90000  # м
theta = np.deg2rad(45)  # рад
phi = np.deg2rad(45)  # рад

# Диапазоны ошибок
delta_r_range = (0, 0)  # смещение по r (всегда 0)
sigma_r_range = (1, 5)  # СКО по r (1-5 м)

delta_theta_range = np.deg2rad((1, 3))  # смещение по theta (1-3 градуса)
sigma_theta_range = np.deg2rad((0.05, 0.15))  # СКО по theta (0.05-0.15 градуса)

delta_phi_range = (0, 0)  # смещение по phi (всегда 0)
sigma_phi_range = np.deg2rad((0.05, 0.15))  # СКО по phi (0.05-0.15 градуса)

# Количество экспериментов
num_experiments = 1000

# Массивы для хранения результатов
delta_x_list, delta_y_list, delta_z_list = [], [], []
sigma_x_list, sigma_y_list, sigma_z_list = [], [], []

# Проведение экспериментов
for _ in range(num_experiments):
    # Генерация случайных ошибок в заданных диапазонах
    delta_r = np.random.uniform(*delta_r_range)
    sigma_r = np.random.uniform(*sigma_r_range)

    delta_theta = np.random.uniform(*delta_theta_range)
    sigma_theta = np.random.uniform(*sigma_theta_range)

    delta_phi = np.random.uniform(*delta_phi_range)
    sigma_phi = np.random.uniform(*sigma_phi_range)

    # Перевод ошибок в декартовы координаты
    delta_x, delta_y, delta_z, sigma_x, sigma_y, sigma_z = spherical_to_cartesian_errors(
        r, theta, phi, delta_r, sigma_r, delta_theta, sigma_theta, delta_phi, sigma_phi
    )

    # Сохранение результатов
    delta_x_list.append(delta_x)
    delta_y_list.append(delta_y)
    delta_z_list.append(delta_z)

    sigma_x_list.append(sigma_x)
    sigma_y_list.append(sigma_y)
    sigma_z_list.append(sigma_z)

# Вычисление минимальных, максимальных и средних значений
def calculate_stats(values):
    return np.min(values), np.max(values), np.mean(values)

delta_x_stats = calculate_stats(delta_x_list)
delta_y_stats = calculate_stats(delta_y_list)
delta_z_stats = calculate_stats(delta_z_list)

sigma_x_stats = calculate_stats(sigma_x_list)
sigma_y_stats = calculate_stats(sigma_y_list)
sigma_z_stats = calculate_stats(sigma_z_list)

# Вывод результатов
print("Смещение в декартовых координатах:")
print(f"Δx: min = {delta_x_stats[0]:.2f} м, max = {delta_x_stats[1]:.2f} м, mean = {delta_x_stats[2]:.2f} м")
print(f"Δy: min = {delta_y_stats[0]:.2f} м, max = {delta_y_stats[1]:.2f} м, mean = {delta_y_stats[2]:.2f} м")
print(f"Δz: min = {delta_z_stats[0]:.2f} м, max = {delta_z_stats[1]:.2f} м, mean = {delta_z_stats[2]:.2f} м")

print("\nСКО в декартовых координатах:")
print(f"σx: min = {sigma_x_stats[0]:.2f} м, max = {sigma_x_stats[1]:.2f} м, mean = {sigma_x_stats[2]:.2f} м")
print(f"σy: min = {sigma_y_stats[0]:.2f} м, max = {sigma_y_stats[1]:.2f} м, mean = {sigma_y_stats[2]:.2f} м")
print(f"σz: min = {sigma_z_stats[0]:.2f} м, max = {sigma_z_stats[1]:.2f} м, mean = {sigma_z_stats[2]:.2f} м")




