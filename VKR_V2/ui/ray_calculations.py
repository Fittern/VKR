import numpy as np
import scipy.signal as signal

def time_domain_calc_rays(signal, sampling_rate, speed_of_sound, num_rays, max_distance, source_velocity=0, receiver_velocity=0):
    num_samples = len(signal)
    time = np.linspace(0, num_samples / sampling_rate, num_samples)
    distances = np.linspace(0, max_distance, num_rays)
    delays = distances / speed_of_sound

    # Вычисление частоты Доплера для каждого луча
    doppler_shifts = [(speed_of_sound + receiver_velocity) / (speed_of_sound + source_velocity) for _ in range(num_rays)]
    
    ray_signals = []
    for delay, doppler_shift in zip(delays, doppler_shifts):
        delay_samples = int(delay * sampling_rate)
        if delay_samples < num_samples:
            delayed_signal = np.zeros_like(signal)
            delayed_signal[delay_samples:] = signal[:-delay_samples] * doppler_shift
            ray_signals.append(delayed_signal)
    
    return ray_signals

def freq_domain_calc_rays(signal, sampling_rate, speed_of_sound, num_rays, max_distance, source_velocity=0, receiver_velocity=0):
    num_samples = len(signal)
    freqs = np.fft.fftfreq(num_samples, d=1/sampling_rate)
    signal_fft = np.fft.fft(signal)
    
    distances = np.linspace(0, max_distance, num_rays)
    delays = distances / speed_of_sound

    # Вычисление частоты Доплера для каждого луча
    doppler_shifts = [(speed_of_sound + receiver_velocity) / (speed_of_sound + source_velocity) for _ in range(num_rays)]

    ray_signals = []
    for delay, doppler_shift in zip(delays, doppler_shifts):
        phase_shift = np.exp(-2j * np.pi * freqs * delay) * doppler_shift
        ray_signal_fft = signal_fft * phase_shift
        ray_signal = np.fft.ifft(ray_signal_fft)
        ray_signals.append(np.real(ray_signal))
    
    return ray_signals

def time_domain_calc(b, s, dist, delay):
    y = signal.fftconvolve(s, b, mode='full')[:len(s)]  # Реакция во временной области
    if delay:
        y = np.concatenate([np.zeros(int(np.ceil(delay))), y[:-int(np.ceil(delay))]])

    y /= (0.8 * np.sqrt(np.pi))  # Нормализация
    y /= dist  # Деление на расстояние
    y_abs = np.abs(np.fft.fft(y))  # Модуль ДПФ реакции
    return y, y_abs

def freq_domain_calc(K, s, dist, fft_size, delay):
    num_segments = len(s) // fft_size  # Количество сегментов
    s = s[:num_segments * fft_size]  # Обрезка s для равномерного деления
    s_segments = np.array(np.split(s, num_segments))  # Разбиение на сегменты
    h = np.fft.ifft(K)  # Импульсный отклик фильтра

    y = []
    for segment in s_segments:
        # Расчет реакции для каждого сегмента
        padded_h = np.pad(h, (0, len(segment) - len(h)), mode='constant')
        yy = np.fft.ifft(np.fft.fft(padded_h) * np.fft.fft(segment))
        y.append(yy[:len(segment)])

    y = np.concatenate(y)
    if delay:
        y = np.concatenate([np.zeros(int(np.ceil(delay))), y[:-int(np.ceil(delay))]])

    y /= (0.8 * np.sqrt(np.pi))
    y /= dist
    y = np.real(y)  # Убедимся, что y вещественное
    y_abs = np.abs(np.fft.fft(y))
    return y, y_abs

def freq_domain_signal_gen(t_sig, segm_num, freq_range, gamma):
    Fs = 32768
    T = 1 / Fs  # Sampling period
    N = int(np.ceil(t_sig / T))  # Number of signal samples

    f = np.arange(0, N) * (Fs / N)  # Frequency vector [0, Fs)
    f0 = f[:N // 2 + 1]  # Frequency vector [0, Fs/2]

    S = np.array([])  # Spectral amplitude vector
    freq = np.array([])  # Frequency vector

    param = 1

    for i in range(segm_num):
        ind1 = np.argmin(np.abs(f0 - freq_range[i, 0]))
        ind2 = np.argmin(np.abs(f0 - freq_range[i, 1]))

        if i == 0:
            tmp = f0[ind1:ind2 + 1]
            start_band_spec = ind1
        else:
            ind1 += 1
            tmp = f0[ind1:ind2 + 1]

        freq = np.append(freq, tmp)

        if tmp.size > 0:
            if i > 0:
                param = S[-1] * ((tmp[0] / 1000) ** gamma[i])
            tmp = param / ((tmp / 1000) ** gamma[i])
            S = np.append(S, tmp)

    fin_band_spec = ind2

    S[0] = S[1]  # Remove Inf at zero frequency

    # Band-pass filter if the band is not initially set from zero or up to Nyquist frequency
    if freq_range[0, 0] != 0 or freq_range[-1, 1] != Fs / 2:
        pre_zeros = np.zeros(start_band_spec)
        post_zeros = np.zeros(len(f0) - fin_band_spec - 1)
        freq = np.concatenate([f0[:start_band_spec], freq, f0[fin_band_spec + 1:]])
        S = np.concatenate([pre_zeros, S, post_zeros])

    # Symmetric mirroring
    if len(f) % 2 == 1:  # If length is odd
        S = np.concatenate([S, np.conj(np.flipud(S[1:]))])
    else:  # If length is even
        S = np.concatenate([S, np.conj(np.flipud(S[1:-1]))])

    # Add random phase noise
    random_phase = np.exp(2j * np.pi * np.random.rand(len(S)))
    S = S * random_phase

    SE = np.fft.ifft(S)  # Apply IFFT
    SE = np.real(SE)  # Ensure the signal is real

    t = np.arange(len(SE)) * T

    return SE, t, freq, S

