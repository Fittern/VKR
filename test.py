import numpy as np
import matplotlib.pyplot as plt

def freq_domain_signal_gen(Fs, t_sig, segm_num, freq_range, gamma):
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

def freq_domain_calc(K, s, dist, fft_size, delay):
    # Buffer the signal into segments
    s_padded = np.pad(s, (0, fft_size - len(s) % fft_size), mode='constant')
    s_segments = s_padded.reshape(-1, fft_size)
    Q = s_segments.shape[0]

    # Calculate the impulse response from the frequency response
    h = np.fft.ifft(K)

    # Response in the frequency domain for each segment
    y_segments = np.zeros_like(s_segments)
    for i in range(Q):
        padded_h = np.pad(h, (0, fft_size - len(h)), mode='constant')
        segment_fft = np.fft.fft(s_segments[i])
        h_fft = np.fft.fft(padded_h)
        yy = np.fft.ifft(h_fft * segment_fft)
        y_segments[i, :] = yy[:fft_size]

    # Concatenate all segments
    y = y_segments.flatten()

    # Add delay if specified
    if delay > 0:
        y = np.concatenate([np.zeros(int(np.ceil(delay))), y])

    # Normalization and distance factor
    y = y / (0.8 * np.sqrt(np.pi))
    y = y / dist

    # Magnitude of the FFT of the response
    y_abs = np.abs(np.fft.fft(y))

    return y, y_abs

def combine_methods():
    # Define filter and FFT characteristics first
    fft_size = 256  # Size of FFT
    K = np.fft.fft(np.random.rand(fft_size))  # Random filter frequency response for demonstration
    dist = 1.5  # Example distance factor
    delay = 5  # Delay in samples

    # Example usage of the freq_domain_signal_gen
    Fs = 1000  # Sampling frequency
    t_sig = 1  # Signal duration in seconds
    segm_num = 2  # Number of segments
    freq_range = np.array([[100, 300], [400, 600]])  # Frequency ranges for each segment
    gamma = np.array([2, 3])  # Decay parameter for each segment

    # Generate signal
    s, t, freq, S = freq_domain_signal_gen(Fs, t_sig, segm_num, freq_range, gamma)

    # Calculate the response
    y, y_abs = freq_domain_calc(K, s, dist, fft_size, delay)

    return y, y_abs

# Run the test and visualize
def test_and_visualize_freq_domain_calc():
    # Define the FFT size and other parameters for testing
    fft_size = 256
    Fs = 1000  # Sampling frequency
    t_sig = 1  # Signal duration in seconds
    segm_num = 3  # Number of frequency segments
    freq_range = np.array([[50, 150], [200, 300], [350, 450]])  # Frequency ranges for each segment
    gamma = np.array([2, 2, 2])  # Decay parameter for each segment
    dist = 1.5  # Example distance factor
    delay = 10  # Delay in samples

    # Generate a signal with specific frequency content
    s, t, freq, S = freq_domain_signal_gen(Fs, t_sig, segm_num, freq_range, gamma)

    # Define a random filter in frequency domain for demonstration
    K = np.fft.fft(np.random.rand(fft_size))

    # Calculate the response using the frequency domain calculation function
    y, y_abs = freq_domain_calc(K, s, dist, fft_size, delay)

    # Visualization
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    
    # Time domain original signal
    axs[0].plot(t, np.real(s), label='Original Signal')
    axs[0].set_title('Original Signal in Time Domain')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].grid(True)
    
    # Time domain processed signal
    axs[1].plot(np.arange(len(y)) / Fs, np.real(y), label='Processed Signal', color='r')
    axs[1].set_title('Processed Signal in Time Domain')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()
    axs[1].grid(True)

    # Frequency domain magnitude of the processed signal
    axs[2].plot(np.fft.fftfreq(len(y), 1/Fs)[:len(y)//2], y_abs[:len(y)//2], label='Magnitude of FFT', color='g')
    axs[2].set_title('Magnitude of FFT of Processed Signal')
    axs[2].set_xlabel('Frequency [Hz]')
    axs[2].set_ylabel('Magnitude')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

# Run the test and visualize the results
test_and_visualize_freq_domain_calc()
