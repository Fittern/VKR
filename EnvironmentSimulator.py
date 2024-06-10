import numpy as np

class EnvironmentSimulator:
    def __init__(self, signal_processor):
        self.signal_processor = signal_processor

    def depth_effect(self, signal, depths, T):
        """
        Моделирует эффекты глубины на сигнал.
        depths - массив значений глубины по времени
        T - период дискретизации сигнала
        """
        C = 1500  # Предполагаемая средняя скорость звука в воде (м/с)
        phase_shifts = 2 * np.pi * np.cumsum(depths) / (C * T)
        depth_modified_signal = signal * np.exp(-1j * phase_shifts)
        return np.real(depth_modified_signal)

    def temperature_effect(self, signal, temperatures, T):
        """
        Моделирует эффекты температуры на сигнал.
        temperatures - массив значений температуры по времени
        T - период дискретизации сигнала
        """
        # Функция для расчета скорости звука в зависимости от температуры
        def sound_speed(temp):
            return 1449.2 + 4.6 * temp - 0.055 * temp**2 + 0.00029 * temp**3

        sound_speeds = sound_speed(temperatures)
        phase_shifts = 2 * np.pi * np.cumsum(sound_speeds) / (sound_speeds * T)
        temperature_modified_signal = signal * np.exp(-1j * phase_shifts)
        return np.real(temperature_modified_signal)

    def doppler(self, signal, fc, v0, phi, v1, psi):
        """
        Моделирует эффект Доплера для сигнала.
        """
        j = 1j
        t = np.arange(len(signal))
        C = 1500  # Скорость звука в воде
        phi = np.radians(phi)
        psi = np.radians(psi)
        fd = fc * (((1 + v1 * np.cos(psi) / C) ** 2) / ((1 - v0 * np.cos(phi) / C) ** 2))
        deltaf = fd - fc
        sd = signal * np.exp(j * 2 * np.pi * deltaf * t)
        return sd


