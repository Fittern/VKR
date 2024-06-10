from SignalProcessor import SignalProcessor, SignalType, NoiseType
import numpy as np
import matplotlib.pyplot as plt

from EnvironmentSimulator import EnvironmentSimulator



def test_environment_effects():
    Fs = 44100  # Частота дискретизации
    sp = SignalProcessor(Fs)
    env_sim = EnvironmentSimulator(sp)
    
    
    # Генерация тестового сигнала
    signal_type = SignalType.TONAL_SIGNAL
    noise_type = NoiseType.PSEUDO_NOISE
    signal_params = (1, 1, 1, [100, 200, 300], [0.1, 0.1, 0.1], [0.05, 0.05], ['gauss', '', 'gauss'])
    noise_params = (1, 1, 300, 1, [0, 0.1, 0.2])
    process_params = {'freq_domain_rays': (np.random.rand(5, 1024), 1, 1024, [0]*5)}

    # Получение сигнала из SignalProcessor
    signal, _, t = sp.process_signals(signal_type, noise_type, signal_params, noise_params, process_params)

    # Подготовка данных по глубине и температуре
    depths = 100 + 50 ** 2 * len(t) * t  # Глубина изменяется от 50 до 150 м
    temperatures = 10 + 5 * np.cos(2 * np.pi * t)  # Температура изменяется от 5 до 15 градусов Цельсия

    # Применение эффектов глубины и температуры
    modified_by_depth = env_sim.depth_effect(signal, depths, 1/Fs)
    modified_by_temperature = env_sim.temperature_effect(signal, temperatures, 1/Fs)

    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, modified_by_depth)
    plt.title('Signal Modified by Depth Changes')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(t, modified_by_temperature)
    plt.title('Signal Modified by Temperature Changes')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()



def test_doppler_effect():
    sp = SignalProcessor(44100)  # Используем частоту дискретизации 44100 Гц
    env_sim = EnvironmentSimulator(sp)
    
    # Параметры для теста
    fc = 1500  # Несущая частота сигнала в Гц
    v0 = 10  # Скорость источника сигнала (м/с)
    phi = 30  # Угол приближения источника к наблюдателю
    v1 = 5  # Скорость наблюдателя (м/с)
    psi = 60  # Угол ухода наблюдателя от источника
    
    # Генерация тестового сигнала
    signal_type = SignalType.TONAL_SIGNAL
    noise_type = NoiseType.PSEUDO_NOISE
    signal_params = (1, 1, 3, [100, 200, 300], [0.1, 0.1, 0.1], [0.05, 0.05], ['gauss', '', 'gauss'])
    noise_params = (1, 1, 300, 1, [0, 0.1, 0.2])
    process_params = {'freq_domain_rays': (np.random.rand(5, 1024), 1, 1024, [0]*5)}

    # Получение сигнала из SignalProcessor
    signal, _, t = sp.process_signals(signal_type, noise_type, signal_params, noise_params, process_params)

    # Применение эффекта Доплера
    doppler_signal = env_sim.doppler(signal, fc, v0, phi, v1, psi)

    # Визуализация результатов
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(t, np.real(doppler_signal))
    plt.title('Signal after Doppler Effect')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_doppler_effect()
    # test_environment_effects()
