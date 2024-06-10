import numpy as np
import matplotlib.pyplot as plt


class SignalGenerator:
    def __init__(self, sample_rate):
        self.Fs = sample_rate

    def df_signal(self, F_low1, F_up1, tau1, F2, tau2, F_low3, F_up3, tau3):
        T = 1 / self.Fs
        t1 = np.arange(0, tau1, T)
        beta1 = (F_up1 - F_low1) / tau1
        s1 = np.cos(2 * np.pi * (beta1 / 2 * t1**2 + F_low1 * t1))

        t2 = np.arange(tau1, tau1 + tau2, T)
        s2 = np.cos(2 * np.pi * F2 * t2)

        t3 = np.arange(0, tau3, T)
        beta3 = (F_up3 - F_low3) / tau3
        s3 = np.cos(2 * np.pi * (beta3 / 2 * t3**2 + F_low3 * t3))
        t3 = t3 + tau1 + tau2

        s = np.concatenate([s1, s2, s3])
        t = np.concatenate([t1, t2, t3])
        return s, t

    def tonal_signal(self, pack_num, pack_dist, pack_imp_num, f_carr, imp_tau, imp_dist, env_shape):
        T = 1 / self.Fs
        s_pack = []

        for i in range(pack_imp_num):
            sigma = 0.15 * imp_tau[i]
            t_imp = np.arange(0, imp_tau[i], T)
            gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_imp - 0.5 * imp_tau[i])**2) / (2 * sigma**2))
            s_imp = np.cos(2 * np.pi * f_carr[i] * t_imp)
            if env_shape[i] == 'gauss':
                s_imp *= gauss / max(gauss)

            if i == pack_imp_num - 1:
                s_space = np.array([])
            else:
                t_space = np.arange(0, imp_dist[i], T)
                s_space = np.zeros(len(t_space))

            s_pack.extend(s_imp)
            s_pack.extend(s_space)

        s = []
        t_pack = np.arange(0, len(s_pack) * T, T)
        t_pack_space = np.arange(0, pack_dist, T)
        s_pack_space = np.zeros(len(t_pack_space))

        for i in range(pack_num):
            s.extend(s_pack)
            s.extend(s_pack_space)

        t = np.arange(0, len(s) * T, T)
        return s, t

    def pfm_signal(self, pack_num, pack_dist, pack_imp_num, f_low, df, ddf, imp_tau, imp_dist, env_shape):
        T = 1 / self.Fs
        s_total = []
        t_total = np.array([])
        current_time = 0

        for j in range(pack_num):
            s_pack = []
            t_pack = np.array([])

            for i in range(pack_imp_num):
                t_imp = np.arange(0, imp_tau[i], T)
                f_up = f_low[i] + df[i] * imp_tau[i] + ddf[i] * (imp_tau[i]**2)
                p = 2
                beta = (f_up - f_low[i]) / (imp_tau[i]**p)
                ft = f_low[i] + beta / (1 + p) * (t_imp**p)
                s_imp = np.cos(2 * np.pi * ft * t_imp)

                if env_shape[i] == 'gauss':
                    sigma = 0.1
                    mu = 0.5 * imp_tau[i]
                    gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_imp - mu)**2) / (2 * sigma**2))
                    s_imp *= gauss / max(gauss)

                s_pack.extend(s_imp)
                t_pack = np.concatenate([t_pack, t_imp + current_time])

                current_time += imp_tau[i]

                if i < pack_imp_num - 1:
                    t_space = np.arange(0, imp_dist[i], T)
                    s_space = np.zeros(len(t_space))
                    s_pack.extend(s_space)
                    t_pack = np.concatenate([t_pack, t_space + current_time])
                    current_time += imp_dist[i]

            s_total.extend(s_pack)
            if j < pack_num - 1:
                t_space_end = np.arange(0, pack_dist, T)
                s_space_end = np.zeros(len(t_space_end))
                s_total.extend(s_space_end)
                t_total = np.concatenate([t_total, t_pack, t_space_end + current_time])
                current_time += pack_dist
            else:
                t_total = np.concatenate([t_total, t_pack])

        return np.array(s_total), t_total

    def lfm_signal(self, pack_num, pack_dist, pack_imp_num, f_low, df, imp_tau, imp_dist, env_shape):
        T = 1 / self.Fs
        s_total = []
        t_total = np.array([])

        for j in range(pack_num):
            s_pack = []
            t_pack = np.array([])
            current_time = 0

            for i in range(pack_imp_num):
                t_imp = np.arange(0, imp_tau[i], T)
                f_up = f_low[i] + df[i] * imp_tau[i]
                beta = (f_up - f_low[i]) / imp_tau[i]
                ft = f_low[i] + beta * t_imp
                s_imp = np.cos(2 * np.pi * ft * t_imp)

                if env_shape == 'gauss':
                    sigma = 0.1
                    mu = 0.5 * imp_tau[i]
                    gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_imp - mu)**2) / (2 * sigma**2))
                    s_imp *= gauss / max(gauss)

                s_pack.extend(s_imp)
                t_pack = np.concatenate([t_pack, t_imp + current_time])

                current_time += imp_tau[i]

                if i < pack_imp_num - 1:
                    t_space = np.arange(0, imp_dist[i], T)
                    s_space = np.zeros(len(t_space))
                    s_pack.extend(s_space)
                    t_pack = np.concatenate([t_pack, t_space + current_time])
                    current_time += imp_dist[i]

            s_total.extend(s_pack)
            if j < pack_num - 1:
                t_space_end = np.arange(0, pack_dist, T)
                s_space_end = np.zeros(len(t_space_end))
                s_total.extend(s_space_end)
                t_total = np.concatenate([t_total, t_pack, t_space_end + current_time])
                current_time += pack_dist
            else:
                t_total = np.concatenate([t_total, t_pack])

        return np.array(s_total), t_total
    
    def hfm_signal(self, pack_num, pack_dist, pack_imp_num, f_low, delta_f, imp_tau, imp_dist, env_shape, mod_type):
        T = 1 / self.Fs
        s_pack = []
        for i in range(pack_imp_num):
            if i == pack_imp_num - 1:
                s_space = []
            else:
                t_space = np.arange(0, imp_dist[i], T)
                s_space = np.zeros(len(t_space))
            t_imp = np.arange(0.01, imp_tau[i] + 0.01, T)
            if mod_type[i] == 1:
                b = delta_f[i] / (imp_tau[i] * f_low[i])
                ft = f_low[i] * (1 - b * t_imp)
            else:
                b = delta_f[i] / (imp_tau[i] * f_low[i])
                ft = f_low[i] * (1 + b * t_imp)
            s_imp = np.cos(2 * np.pi * ft * t_imp)
            if env_shape == 'gauss':
                sigma = 0.1
                mu = 0
                gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_imp - 0.5 * imp_tau[i] - mu)**2) / (2 * sigma**2))
                s_imp *= gauss / max(gauss)
            s_pack.extend(s_imp)
            s_pack.extend(s_space)
        s = []
        t_pack_space = np.arange(0, pack_dist, T)
        s_pack_space = np.zeros(len(t_pack_space))
        for i in range(pack_num):
            s.extend(s_pack)
            s.extend(s_pack_space)
        t = np.arange(0, len(s) * T, T)
        return s, t


if __name__ == "__main__":
    sg = SignalGenerator(44100)
    
    # # Тестирование генерации PFM сигнала
    # s, t = sg.pfm_signal(
    #     pack_num=1,
    #     pack_dist=0.5,
    #     pack_imp_num=2,
    #     f_low=[100, 200],
    #     df=[10, 20],
    #     ddf=[0.5, 0.5],
    #     imp_tau=[0.1, 0.1],
    #     imp_dist=[0.05, 0.05],
    #     env_shape=['gauss', 'gauss']
    # )
    
    # Тестирование генерации LFM сигнала
    # s, t = sg.lfm_signal(
    #     pack_num=1,
    #     pack_dist=0.5,
    #     pack_imp_num=3,
    #     f_low=[100, 200, 300],
    #     df=[10, 20, 30],
    #     imp_tau=[0.1, 0.1, 0.1],
    #     imp_dist=[0.05, 0.05],
    #     env_shape='gauss'
    # )
    
    # Тестирование генерации HFM сигнала
    s, t = sg.hfm_signal(
        pack_num=1,
        pack_dist=0.5,
        pack_imp_num=2,
        f_low=[100, 100],
        delta_f=[10, 10,],
        imp_tau=[0.1, 0.1],
        imp_dist=[0.05, 0.05],
        env_shape='gauss',
        mod_type=[1, -1]
    )
    
    # Вывод первых 10 элементов каждого сигнала для просмотра результатов
    # print("PFM signal samples:", pfm_s[:10])
    # print("LFM signal samples:", lfm_s)
    # print("HFM signal samples:", hfm_s[:10])
    
    # Создание экземпляра класса и генерация сигнала
    # sg = SignalGenerator(44100)
    # s, t = sg.pfm_signal(pack_num=1, pack_dist=1, pack_imp_num=3,
    #                      f_low=[100, 200, 300], df=[10, 20, 30], ddf=[0.5, 0.5, 0.5],
    #                      imp_tau=[0.1, 0.1, 0.1], imp_dist=[0.05, 0.05], 
    #                      env_shape=['gauss', '', 'gauss'])
    
    # s, t = sg.tonal_signal(1, 1, 3, [100, 200, 300], [0.2, 0.2, 0.2], [0.1, 0.1], ['rect', 'gauss', 'rect'])
    # s, t = sg.df_signal(300, 500, 0.1, 100, 0.05, 100, 800, 0.1)
    # Визуализация сигнала
    plt.figure(figsize=(10, 10))
    plt.plot(t, s)
    plt.title('HFM Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    
    # sg = SignalGenerator(44100)
    
    # # Визуализация сигнала
    # plt.figure(figsize=(10, 4))
    # plt.plot(t_df, s_df)
    # plt.title('DF Signal')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()