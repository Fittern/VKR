from NoiseGenerator import NoiseGenerator
from SignalGenerator import SignalGenerator

import numpy as np
from enum import Enum, auto
import scipy.signal as signal
import matplotlib.pyplot as plt


class SignalType(Enum):
    DF_SIGNAL = auto()
    TONAL_SIGNAL = auto()
    PFM_SIGNAL = auto()
    LFM_SIGNAL = auto()
    HFM_SIGNAL = auto()

class NoiseType(Enum):
    PSEUDO_NOISE = auto()
    WHITE_NOISE = auto()

class SignalProcessor:
    def __init__(self, sample_rate):
        self.Fs = sample_rate
        self.signal_generator = SignalGenerator(sample_rate)
        self.noise_generator = NoiseGenerator(sample_rate)

    def freq_domain_calc(self, K, s, dist, fft_size, delay):
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

    def freq_domain_calc_rays(self, K, s, dist, fft_size, delay):
        rays_num = K.shape[0]
        y_rays = np.zeros((rays_num, len(s) + int(np.ceil(max(delay)))))
        y_rays_abs = np.zeros((rays_num, len(s) + int(np.ceil(max(delay)))))

        for i in range(rays_num):
            y_ray, y_ray_abs = self.freq_domain_calc(K[i], s, dist, fft_size, delay[i])
            y_rays[i, :len(y_ray)] = y_ray
            y_rays_abs[i, :len(y_ray_abs)] = y_ray_abs

        y_rays_all = np.sum(y_rays, axis=0)
        y_rays_all_abs = np.sum(y_rays_abs**2, axis=0)
        return y_rays, y_rays_all, y_rays_abs, y_rays_all_abs

    def freq_domain_signal_gen(self, t_sig, segm_num, freq_range, gamma):
        T = 1 / self.Fs  # Sampling period
        N = int(np.ceil(t_sig / T))  # Number of signal samples

        f = np.arange(0, N) * (self.Fs / N)  # Frequency vector [0, Fs)
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
        if freq_range[0, 0] != 0 or freq_range[-1, 1] != self.Fs / 2:
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

    def time_domain_calc(self, b, s, dist, delay):
        y = signal.fftconvolve(s, b, mode='full')[:len(s)]  # Реакция во временной области
        if delay:
            y = np.concatenate([np.zeros(int(np.ceil(delay))), y[:-int(np.ceil(delay))]])

        y /= (0.8 * np.sqrt(np.pi))  # Нормализация
        y /= dist  # Деление на расстояние
        y_abs = np.abs(np.fft.fft(y))  # Модуль ДПФ реакции
        return y, y_abs

    def time_domain_calc_rays(self, b, s, dist, delay):
        rays_num = b.shape[0]
        y_rays = np.zeros((rays_num, len(s) + int(np.ceil(max(delay)))))
        y_rays_abs = np.zeros((rays_num, len(s) + int(np.ceil(max(delay)))))

        for i in range(rays_num):
            y_ray, y_ray_abs = self.time_domain_calc(b[i], s, dist, delay[i])
            y_rays[i, :len(y_ray)] = y_ray
            y_rays_abs[i, :len(y_ray_abs)] = y_ray_abs

        y_rays_all = np.sum(y_rays, axis=0)
        y_rays_all_abs = np.sum(y_rays_abs, axis=0)
        return y_rays, y_rays_all, y_rays_abs, y_rays_all_abs

    def process_signals1(self, signal_type, noise_type, signal_params, noise_params):
        # Generate signal
        if signal_type == SignalType.DF_SIGNAL:
            s, t = self.signal_generator.df_signal(*signal_params)
        elif signal_type == SignalType.TONAL_SIGNAL:
            s, t = self.signal_generator.tonal_signal(*signal_params)
        elif signal_type == SignalType.PFM_SIGNAL:
            s, t = self.signal_generator.pfm_signal(*signal_params)
        elif signal_type == SignalType.LFM_SIGNAL:
            s, t = self.signal_generator.lfm_signal(*signal_params)
        elif signal_type == SignalType.HFM_SIGNAL:
            s, t = self.signal_generator.hfm_signal(*signal_params)

        # Generate noise
        if noise_type == NoiseType.PSEUDO_NOISE:
            n, _ = self.noise_generator.pseudo_noise(*noise_params)
        elif noise_type == NoiseType.WHITE_NOISE:
            n, _ = self.noise_generator.white_noise(*noise_params[:2])  # Assumes the correct number of parameters

        # Ensure s and n are the same length
        min_length = min(len(s), len(n))
        s = s[:min_length]
        n = n[:min_length]
        t = t[:min_length]

        # Combine signal and noise
        return s + n, t
    
    def process_signals(self, signal_type, noise_type, signal_params, noise_params, process_params):
        combined_signal, t = self.process_signals1(signal_type, noise_type, signal_params, noise_params);
        # Check what processing parameters are provided and process accordingly

        if 'freq_domain_rays' in process_params:
            K, dist, fft_size, delays = process_params['freq_domain_rays']
            y_rays, y_rays_all, y_rays_abs, y_rays_all_abs = self.freq_domain_calc_rays(K, combined_signal, dist, fft_size, delays)
            y = y_rays_all
            y_abs = y_rays_all_abs
        elif 'time_domain_rays' in process_params:
            b, dist, delays = process_params['time_domain_rays']
            y_rays, y_rays_all, y_rays_abs, y_rays_all_abs = self.time_domain_calc_rays(b, combined_signal, dist, delays)
            y = y_rays_all
            y_abs = y_rays_all_abs
        # elif 'freq_domain_signal' in process_params:
        #     t_sig, segm_num, freq_range, gamma = process_params['freq_domain_signal']
        #     y, t, f, S = self.freq_domain_signal_gen(t_sig, segm_num, freq_range, gamma)
        #     y_abs = np.abs(np.fft.fft(y))

        # Visualize the output
        # plt.figure(figsize=(10, 4))
        # plt.plot(np.linspace(0, len(y) / self.Fs, num=len(y), endpoint=False), y)
        # plt.title('Processed Signal Output')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.show()

        return y, y_abs, t


if __name__ == "__main__":
    # Initialize SignalProcessor with sample rate
    sp = SignalProcessor(44100)

    # Define common parameters for signal and noise
    signal_params = (1, 1, 3, [100, 200, 300], [0.1, 0.1, 0.1], [0.05, 0.05], ['gauss', 'gauss', 'gauss'])
    noise_params = (2, 2, 500, 3, [0, 0.1, 0.2])

    # Test for freq_domain_rays
    process_params_freq_rays = {
        'freq_domain_rays': (np.random.rand(5, 1024), 1, 1024, [0.1]*5)
    }
    y_freq_rays, y_abs_freq_rays, _ = sp.process_signals(SignalType.TONAL_SIGNAL, NoiseType.PSEUDO_NOISE, signal_params, noise_params, process_params_freq_rays)

    # Test for time_domain_rays
    process_params_time_rays = {
        'time_domain_rays': (np.random.rand(5, 32), 10, [0.1]*5)
    }
    y_time_rays, y_abs_time_rays, _ = sp.process_signals(SignalType.TONAL_SIGNAL, NoiseType.PSEUDO_NOISE, signal_params, noise_params, process_params_time_rays)

    # # Test for freq_domain_signal
    # process_params_freq_signal = {
    #     'freq_domain_signal': (0.5, 3, np.array([[0, 0.1], [0.15, 0.3], [0.35, 0.5]]), [1, 2, 1.5])
    # }
    # y_freq_signal, y_abs_freq_signal, _ = sp.process_signals(SignalType.TONAL_SIGNAL, NoiseType.PSEUDO_NOISE, signal_params, noise_params, process_params_freq_signal)

    s = sp.process_signals1(SignalType.TONAL_SIGNAL, NoiseType.PSEUDO_NOISE, signal_params, noise_params)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(s)
    plt.title('TONAL_SIGNAL whith PSEUDO_NOISE')
    plt.subplot(1, 2, 2)
    plt.plot(y_time_rays)
    plt.title('Signal after time_domain_rays')
    plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.plot(y_freq_rays)
    # plt.title('Signal after time_domain_rays')
    # plt.show()

    # Output results for review
    print("Frequency Domain Rays Output:", y_freq_rays[:10])
    print("Time Domain Rays Output:", y_time_rays[:10])
    # print("Frequency Domain Signal Output:", y_freq_signal[:10])


