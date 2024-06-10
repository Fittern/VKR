import numpy as np
import scipy.signal as signal

class NoiseGenerator:
    def __init__(self, sample_rate):
        self.Fs = sample_rate

    def pseudo_noise(self, pack_num, pack_dist, f_carr, tau, freq):
        T = 1 / self.Fs
        N = int(np.ceil(tau / T))
        t = np.arange(N) * T
        jumps_num = np.random.randint(2, N)  # количество скачков
        
        ind = np.sort(np.random.choice(np.arange(1, N), jumps_num, replace=False))
        ind = np.append(np.append(0, ind), N)  # добавляем начальную и конечную точки
        
        A = 1
        phi = 2 * np.pi * np.random.rand(len(ind) - 1) - np.pi  # случайные фазы

        s_pack = []
        for i in range(len(ind) - 1):
            t_segment = t[ind[i]:ind[i + 1]]
            s_pack.extend(A * np.sin(2 * np.pi * f_carr * t_segment + phi[i]))

        # Design FIR filter
        n = 1000  # filter order
        if freq[0] == 0:
            freq = np.array(freq[1:])
            b = signal.firwin(n, freq / (0.5 * self.Fs), pass_zero=False)
        else:
            b = signal.firwin(n, freq / (0.5 * self.Fs), pass_zero=True)

        s_pack = signal.lfilter(b, 1, s_pack)  # Apply the FIR filter

        # Prepare the output with spaces between packets
        s_total = []
        t_total = np.array([])
        current_time = 0
        for i in range(pack_num):
            s_total.extend(s_pack)
            t_pack = np.arange(len(s_pack)) * T + current_time
            if i < pack_num - 1:
                s_space = np.zeros(int(pack_dist / T))
                s_total.extend(s_space)
                t_space = np.arange(len(s_space)) * T + t_pack[-1] + T
                t_total = np.concatenate([t_total, t_pack, t_space])
                current_time = t_total[-1] + T
            else:
                t_total = np.concatenate([t_total, t_pack])
        
        return np.array(s_total), t_total

    def white_noise(self, t_sig, sigma):
        T = 1 / self.Fs
        N = int(np.floor((t_sig / T) + 0.5))
        s = sigma * np.random.randn(N)
        t = np.arange(0, N) * T
        return s, t

if __name__ == "__main__":
    # Example usage
    ng = NoiseGenerator(44100)
    s_pseudo, t_pseudo = ng.pseudo_noise(pack_num=1, pack_dist=1, f_carr=300, tau=1, freq=[0, 0.1, 0.2])
    s_white, t_white = ng.white_noise(t_sig=1, sigma=0.5)
    
    # Visualization (if needed, uncomment these lines)
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    plt.figure(figsize=(10, 10))

    plt.plot(t_pseudo, s_pseudo)
    plt.title('Pseudo Noise Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 2)
    plt.plot(t_white, s_white)
    plt.title('White Noise Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    