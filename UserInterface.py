import tkinter as tk
import json
import numpy as np
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from SignalProcessor import SignalProcessor, SignalType, NoiseType

class UserInterface:
    def __init__(self, master):
        self.master = master
        master.title("Signal and Noise Processing Simulator")

        self.signal_processor = SignalProcessor(44100)  # Создание экземпляра SignalProcessor

        # Элементы управления для выбора типа сигнала и шума
        self.signal_type_var = tk.StringVar()
        self.noise_type_var = tk.StringVar()
        self.signal_type_combo = ttk.Combobox(master, textvariable=self.signal_type_var, values=[e.name for e in SignalType], state="readonly")
        self.noise_type_combo = ttk.Combobox(master, textvariable=self.noise_type_var, values=[e.name for e in NoiseType], state="readonly")

        self.signal_type_combo.bind("<<ComboboxSelected>>", self.update_signal_params_ui)
        self.noise_type_combo.bind("<<ComboboxSelected>>", self.update_noise_params_ui)

        # Кнопка для генерации сигнала и шума
        self.generate_button = tk.Button(master, text="Generate Signal and Noise", command=self.generate_signal_and_noise)

        # Место для отображения графика
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Расположение элементов управления
        self.signal_type_combo.pack()
        self.noise_type_combo.pack()
        self.generate_button.pack()

        # Панели для параметров сигнала и шума
        self.signal_params_frame = tk.Frame(master)
        self.noise_params_frame = tk.Frame(master)
        self.signal_params_frame.pack()
        self.noise_params_frame.pack()

        # Параметры по умолчанию для сигнала и шума
        self.default_signal_params = {
            'DF_SIGNAL': (100, 500, 0.1, 300, 0.05, 500, 1000, 0.1),
            'TONAL_SIGNAL': (1, 1, 3, [100, 200, 300], [0.1, 0.1, 0.1], [0.05, 0.05], ['gauss', 'gauss', 'gauss']),
            'PFM_SIGNAL': (1, 1, 3, [100, 200, 300], [10, 20, 30], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.05, 0.05, 0.05], ['gauss', 'gauss', 'gauss']),
            'LFM_SIGNAL': (1, 1, 3, [100, 200, 300], [10, 20, 30], [0.1, 0.1, 0.1], [0.05, 0.05], 'gauss'),
            'HFM_SIGNAL': (1, 1, 3, [100, 200, 300], [10, 20, 30], [0.1, 0.1, 0.1], [0.05, 0.05, 0.05], 'gauss', [1, -1, 1])
        }
        self.default_noise_params = {
            'PSEUDO_NOISE': (10, 0, 300, 1, [0, 0.1, 0.2]),
            'WHITE_NOISE': (100, 0.2)
        }

    def update_signal_params_ui(self, event=None):
        # Очистка старых виджетов
        for widget in self.signal_params_frame.winfo_children():
            widget.destroy()

        # Получаем выбранный тип сигнала
        signal_type = self.signal_type_var.get()

        # Названия параметров для разных типов сигналов
        param_labels = {
            'DF_SIGNAL': ["F_low1", "F_up1", "tau1", "F2", "tau2", "F_low3", "F_up3", "tau3"],
            'TONAL_SIGNAL': ["pack_num", "pack_dist", "pack_imp_num", "f_carr", "imp_tau", "imp_dist", "env_shape"],
            'PFM_SIGNAL': ["pack_num", "pack_dist", "pack_imp_num", "f_low", "df", "ddf", "imp_tau", "imp_dist", "env_shape"],
            'LFM_SIGNAL': ["pack_num", "pack_dist", "pack_imp_num", "f_low", "df", "imp_tau", "imp_dist", "env_shape"],
            'HFM_SIGNAL': ["pack_num", "pack_dist", "pack_imp_num", "f_low", "delta_f", "imp_tau", "imp_dist", "env_shape", "mod_type"]
        }

        # Создаем виджеты для ввода параметров для выбранного типа сигнала
        if signal_type:
            params = self.default_signal_params[signal_type]
            for param, value in zip(param_labels[signal_type], params):
                label = tk.Label(self.signal_params_frame, text=f"{param}:")
                entry = tk.Entry(self.signal_params_frame)
                entry.insert(0, str(value))
                label.pack(side=tk.LEFT)
                entry.pack(side=tk.LEFT)

    def update_noise_params_ui(self, event=None):
        # Очистка старых виджетов
        for widget in self.noise_params_frame.winfo_children():
            widget.destroy()
    
        # Получаем выбранный тип шума
        noise_type = self.noise_type_var.get()
    
        # Названия параметров для разных типов шумов
        param_labels = {
            'PSEUDO_NOISE': ["pack_num", "pack_dist", "f_carr", "tau", "freq"],
            'WHITE_NOISE': ["t_sig", "sigma"]
        }
    
        # Создаем виджеты для ввода параметров для выбранного типа шума
        if noise_type:
            params = self.default_noise_params[noise_type]
            for param, value in zip(param_labels[noise_type], params):
                label = tk.Label(self.noise_params_frame, text=f"{param}:")
                entry = tk.Entry(self.noise_params_frame)
                entry.insert(0, str(value))
                label.pack(side=tk.LEFT)
                entry.pack(side=tk.LEFT)

    def generate_signal_and_noise(self):
        # Сбор параметров сигнала из пользовательского интерфейса
        signal_params = [entry.get() for entry in self.signal_params_frame.winfo_children() if isinstance(entry, tk.Entry)]
        noise_params = [entry.get() for entry in self.noise_params_frame.winfo_children() if isinstance(entry, tk.Entry)]
    
        # Преобразование параметров из строк в необходимые типы данных
        signal_params = self.convert_params(signal_params, self.signal_type_var.get())
        noise_params = self.convert_params(noise_params, self.noise_type_var.get())
    
        # Получение типа сигнала и шума
        signal_type = SignalType[self.signal_type_var.get()]
        noise_type = NoiseType[self.noise_type_var.get()]

        sp = SignalProcessor(44100)
    
        # Переменные для результатов, инициализируем пустыми массивами для безопасности
        y = []
        y_abs = []
        t = []
    
        try:
            # Обработка сигнала и шума
            y, y_abs, t = self.signal_processor.process_signals(signal_type, noise_type, signal_params, noise_params, process_params={'freq_domain_rays': (np.random.rand(5, 1024), 1, 1024, [0]*5)})
        except Exception as e:
            print(f"Error processing signals: {e}")
            return  # Возврат из функции, если произошла ошибка
    
        # Отображение результатов на графике
        self.ax.clear()
        self.ax.plot(t, y)  # Добавляем время t к графику для корректного отображения
        self.ax.set_title(f"Frequency Domain Rays - {signal_type.name} with {noise_type.name}")
        self.canvas.draw()

    def convert_params(self, params, type_name):
        """
        Конвертирует параметры из строки в соответствующие типы данных, основываясь на типе сигнала или шума.
        Это упрощенный пример конвертации. Реальная конвертация может потребовать более сложной логики в зависимости от параметров.
        """
        converted_params = []
        for param in params:
            try:
                # Попытка интерпретировать параметр как JSON-форматированную строку
                param = json.loads(param.replace("'", '"'))
            except json.JSONDecodeError:
                # Если не удалось преобразовать JSON, попробуем просто преобразовать в число, если возможно
                try:
                    param = float(param) if '.' in param else int(param)
                except ValueError:
                    pass  # оставляем param без изменений, если это не число

            converted_params.append(param)

        return converted_params


if __name__ == "__main__":
    root = tk.Tk()
    app = UserInterface(root)
    root.mainloop()