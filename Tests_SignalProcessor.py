import numpy as np
import matplotlib.pyplot as plt
from SignalProcessor import SignalProcessor, SignalType, NoiseType

def test_freq_domain_calc():
    sp = SignalProcessor(44100)  # Создаем экземпляр с частотой дискретизации 44100 Гц
    # Генерация тестового сигнала
    t = np.linspace(0, 1, 44100)
    s = np.sin(2 * np.pi * 440 * t)  # Тональный сигнал 440 Гц

    # Убедимся, что длина s кратна fft_size
    fft_size = 1024
    s = s[:len(s) - len(s) % fft_size]  # Обрезаем сигнал до кратного fft_size размера
    
    # Генерация тестового фильтра
    K = np.fft.fft(np.hanning(1000), n=fft_size)  # Подгоняем размер фильтра под fft_size
    dist = 1
    delay = 0

    # Вызов функции обработки
    y, y_abs = sp.freq_domain_calc(K, s, dist, fft_size, delay)

    # Визуализация результатов
    plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    plt.plot(y)
    plt.title('Output Signal')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

    # plt.subplot(1, 2, 2)
    # plt.plot(20 * np.log10(y_abs + np.finfo(float).eps))  # Добавляем eps для избежания логарифма нуля
    # plt.title('Magnitude Spectrum of the Output')
    # plt.xlabel('Frequency (Bins)')
    # plt.ylabel('Magnitude (dB)')
    # plt.grid(True)
    # plt.tight_layout()
    plt.show()


def test_freq_domain_calc_rays():
    sp = SignalProcessor(44100)
    t = np.linspace(0, 1, 44100)
    s = np.sin(2 * np.pi * 400 * t)
    
    # Убедимся, что длина s кратна fft_size
    fft_size = 2048
    s = s[:len(s) - len(s) % fft_size]  # Обрезаем сигнал до кратного fft_size размера
    
    # Создание нескольких фильтров
    K = np.array([np.fft.fft(np.hanning(fft_size), n=fft_size) for _ in range(2)])
    dist = 3
    delays = np.array([0, 5])  # Добавление задержек

    # Вызов функции обработки
    y_rays, y_rays_all, y_rays_abs, y_rays_all_abs = sp.freq_domain_calc_rays(K, s, dist, fft_size, delays)
    
    # Визуализация суммарной реакции
    plt.figure(figsize=(10, 10))
    plt.plot(y_rays_all)
    plt.title('Combined Response from Rays')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def test_process_signals():
    # Initialize SignalProcessor with a given sample rate
    sp = SignalProcessor(44100)

    # Signal generation parameters
    signal_params = {
        SignalType.DF_SIGNAL: (100, 500, 0.1, 300, 0.05, 500, 1000, 0.1),
        SignalType.TONAL_SIGNAL: (4, 1, 3, [100, 200, 300], [0.1, 0.1, 0.1], [0.05, 0.05], ['gauss', 'gauss', 'gauss']),
        SignalType.PFM_SIGNAL: (2, 1, 3, [100, 200, 300], [10, 20, 30], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.05, 0.05, 0.05], ['gauss', 'gauss', 'gauss']),
        SignalType.LFM_SIGNAL: (3, 1, 3, [100, 200, 300], [10, 20, 30], [0.1, 0.1, 0.1], [0.05, 0.05], 'gauss'),
        SignalType.HFM_SIGNAL: (6, 1, 3, [100, 200, 300], [10, 20, 30], [0.1, 0.1, 0.1], [0.05, 0.05, 0.05], 'gauss', [1, -1, 1])
    }

    # Noise generation parameters
    noise_params = {
        NoiseType.PSEUDO_NOISE: (10, 0, 300, 1, [0, 0.1, 0.2]),
        NoiseType.WHITE_NOISE: (100, 0.2)
    }

    # Processing parameters for different methods
    process_params = {
        'freq_domain_rays': (np.random.rand(5, 1024), 1, 1024, [0]*5),
        'time_domain_rays': (np.random.rand(5, 256), 1, [0]*5),
        # 'freq_domain_signal': (0.5, 3, np.array([[0, 0.1], [0.15, 0.3], [0.35, 0.5]]), [1, 2, 1.5])
    }

    # Test each combination of signal and noise types
    for signal_type in SignalType:
        for noise_type in NoiseType:
            print(f"Testing {signal_type.name} with {noise_type.name}")
            s_params = signal_params[signal_type]
            n_params = noise_params[noise_type]
            y, y_abs, _ = sp.process_signals(signal_type, noise_type, s_params, n_params, process_params)

            # plt.figure(figsize=(12, 6))
            # plt.subplot(2, 1, 1)
            plt.figure(figsize=(10, 10))
            plt.plot(y)
            plt.title(f'Processed Signal - {signal_type.name} with {noise_type.name}')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')

            # plt.subplot(2, 1, 2)
            # plt.plot(y_abs)
            # plt.title(f'Spectrum of Processed Signal - {signal_type.name} with {noise_type.name}')
            # plt.xlabel('Frequency Bin')
            # plt.ylabel('Magnitude')

            # plt.tight_layout()
            plt.show()
            
if __name__ == "__main__":
    test_freq_domain_calc()
    test_freq_domain_calc_rays()
    test_process_signals()
    # Example usage

    sp = SignalProcessor(44100)
    f_carr = 300
    tau = 1
    freq = [0, 0.1, 0.2]
    s, t, f, S = sp.freq_domain_signal_gen(tau, 1, np.array([[0, 5000]]), [0.1])
    print("Generated frequency domain signal.")


#     EnvironmentSimulator

# doppler_effect(frequency, velocity)
# depth_temperature_effect(depth, temperature, signal)
# UserInterface

# setup_parameters()
# display_results(results)
