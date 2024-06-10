import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QVBoxLayout, QWidget, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QFormLayout, QCheckBox, QLineEdit, QGridLayout
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from signal_generators import SignalGenerator
from noise_generators import NoiseGenerator
from ray_calculations import time_domain_calc_rays, freq_domain_calc_rays, time_domain_calc, freq_domain_calc, freq_domain_signal_gen
from enum import Enum, auto


class SignalType(Enum):
    DF_SIGNAL = auto()
    TONAL_SIGNAL = auto()
    PFM_SIGNAL = auto()
    LFM_SIGNAL = auto()
    HFM_SIGNAL = auto()

class NoiseType(Enum):
    PSEUDO_NOISE = auto()
    WHITE_NOISE = auto()

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Hydroacoustic Signal Simulator")
        self.setGeometry(100, 100, 1000, 800)

        # Создание меню
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')

        # Добавление действий в меню
        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        # Создание центрального виджета
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        layout = QVBoxLayout(centralWidget)

        # Поле для графиков
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)

        param_layout = QHBoxLayout()
        layout.addLayout(param_layout)

        left_layout = QVBoxLayout()
        param_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()
        param_layout.addLayout(right_layout)

        # Параметры сигнала
        self.signal_type_combo = QComboBox(self)
        self.signal_type_combo.addItems(["DF Signal", "Tonal Signal", "PFM Signal", "LFM Signal", "HFM Signal"])
        self.signal_type_combo.currentTextChanged.connect(self.update_signal_params)
        left_layout.addWidget(self.signal_type_combo)

        self.sampling_rate_spinbox = QSpinBox(self)  # Создание sampling_rate_spinbox
        self.sampling_rate_spinbox.setRange(100, 100000)
        self.sampling_rate_spinbox.setValue(1000)
        self.sampling_rate_spinbox.setSuffix(" Hz")
        left_layout.addWidget(self.sampling_rate_spinbox)  # Добавление sampling_rate_spinbox в layout

        self.signal_params_layout = QFormLayout()
        left_layout.addLayout(self.signal_params_layout)
        self.update_signal_params()

        # Параметры шума
        self.noise_checkbox = QCheckBox("Add Noise", self)
        self.noise_checkbox.stateChanged.connect(self.toggle_noise_params)
        right_layout.addWidget(self.noise_checkbox)

        self.noise_params_layout = QFormLayout()
        self.noise_type_combo = QComboBox(self)
        self.noise_type_combo.addItems(["Pseudo Noise", "White Noise"])
        self.noise_type_combo.currentTextChanged.connect(self.update_noise_params)
        self.noise_params_layout.addRow("Noise Type:", self.noise_type_combo)

        self.noise_level_spinbox = QDoubleSpinBox(self)
        self.noise_level_spinbox.setRange(0, 1)
        self.noise_level_spinbox.setValue(0.1)
        self.noise_params_layout.addRow("Noise Level:", self.noise_level_spinbox)

        right_layout.addLayout(self.noise_params_layout)
        self.update_noise_params()

        # Спинбоксы для физических параметров
        self.physical_params_layout = QFormLayout()
        self.depth_spinbox = QDoubleSpinBox(self)
        self.depth_spinbox.setRange(0, 11000)
        self.depth_spinbox.setValue(0)
        self.depth_spinbox.setSuffix(" m")
        self.physical_params_layout.addRow("Depth:", self.depth_spinbox)

        self.temperature_spinbox = QDoubleSpinBox(self)
        self.temperature_spinbox.setRange(-2, 40)
        self.temperature_spinbox.setValue(15)
        self.temperature_spinbox.setSuffix(" °C")
        self.physical_params_layout.addRow("Temperature:", self.temperature_spinbox)

        self.salinity_spinbox = QDoubleSpinBox(self)
        self.salinity_spinbox.setRange(0, 40)
        self.salinity_spinbox.setValue(35)
        self.salinity_spinbox.setSuffix(" PSU")
        self.physical_params_layout.addRow("Salinity:", self.salinity_spinbox)

        self.noise_threshold_spinbox = QDoubleSpinBox(self)
        self.noise_threshold_spinbox.setRange(0, 1)
        self.noise_threshold_spinbox.setValue(0.5)
        self.noise_threshold_spinbox.setSuffix(" Threshold")
        self.physical_params_layout.addRow("Noise Threshold:", self.noise_threshold_spinbox)

        left_layout.addLayout(self.physical_params_layout)

        # Кнопка для генерации сигнала
        self.generate_button = QPushButton("Generate Signal", self)
        self.generate_button.clicked.connect(self.generate_signal)
        left_layout.addWidget(self.generate_button)

        # Кнопка для расчетов лучей
        self.calculate_button = QPushButton("Calculate Rays", self)
        self.calculate_button.clicked.connect(self.calculate_rays)
        left_layout.addWidget(self.calculate_button)

        # Кнопки для режимов работы
        self.noise_detection_button = QPushButton("Noise Detection Mode", self)
        self.noise_detection_button.clicked.connect(self.noise_detection_mode)
        left_layout.addWidget(self.noise_detection_button)

        self.hydroacoustic_mode_button = QPushButton("Hydroacoustic Mode", self)
        self.hydroacoustic_mode_button.clicked.connect(self.hydroacoustic_mode)
        left_layout.addWidget(self.hydroacoustic_mode_button)

        self.show()



    def update_signal_params(self):
        for i in reversed(range(self.signal_params_layout.count())): 
            self.signal_params_layout.itemAt(i).widget().setParent(None)

        signal_type = self.signal_type_combo.currentText()
        if signal_type == "DF Signal":
            self.add_param(self.signal_params_layout, "F_low1", 20)
            self.add_param(self.signal_params_layout, "F_up1", 50)
            self.add_param(self.signal_params_layout, "tau1", 1)
            self.add_param(self.signal_params_layout, "F2", 100)
            self.add_param(self.signal_params_layout, "tau2", 2)
            self.add_param(self.signal_params_layout, "F_low3", 150)
            self.add_param(self.signal_params_layout, "F_up3", 200)
            self.add_param(self.signal_params_layout, "tau3", 3)
        elif signal_type == "Tonal Signal":
            self.add_param(self.signal_params_layout, "pack_num", 3)
            self.add_param(self.signal_params_layout, "pack_dist", 1)
            self.add_param(self.signal_params_layout, "pack_imp_num", 2)
            self.add_param(self.signal_params_layout, "f_carr", "50,100")
            self.add_param(self.signal_params_layout, "imp_tau", "1,1.5")
            self.add_param(self.signal_params_layout, "imp_dist", "0.5,0.5")
            self.add_param(self.signal_params_layout, "env_shape", "gauss,gauss")
        elif signal_type == "PFM Signal":
            self.add_param(self.signal_params_layout, "pack_num", 3)
            self.add_param(self.signal_params_layout, "pack_dist", 1)
            self.add_param(self.signal_params_layout, "pack_imp_num", 2)
            self.add_param(self.signal_params_layout, "f_low", "20,30")
            self.add_param(self.signal_params_layout, "df", "10,15")
            self.add_param(self.signal_params_layout, "ddf", "5,10")
            self.add_param(self.signal_params_layout, "imp_tau", "1,1.5")
            self.add_param(self.signal_params_layout, "imp_dist", "0.5,0.5")
            self.add_param(self.signal_params_layout, "env_shape", "gauss,gauss")
        elif signal_type == "LFM Signal":
            self.add_param(self.signal_params_layout, "pack_num", 3)
            self.add_param(self.signal_params_layout, "pack_dist", 1)
            self.add_param(self.signal_params_layout, "pack_imp_num", 2)
            self.add_param(self.signal_params_layout, "f_low", "20,30")
            self.add_param(self.signal_params_layout, "df", "10,15")
            self.add_param(self.signal_params_layout, "imp_tau", "1,1.5")
            self.add_param(self.signal_params_layout, "imp_dist", "0.5,0.5")
            self.add_param(self.signal_params_layout, "env_shape", "gauss,gauss")
        elif signal_type == "HFM Signal":
            self.add_param(self.signal_params_layout, "pack_num", 3)
            self.add_param(self.signal_params_layout, "pack_dist", 1)
            self.add_param(self.signal_params_layout, "pack_imp_num", 2)
            self.add_param(self.signal_params_layout, "f_low", "20,30")
            self.add_param(self.signal_params_layout, "delta_f", "10,15")
            self.add_param(self.signal_params_layout, "imp_tau", "1,1.5")
            self.add_param(self.signal_params_layout, "imp_dist", "0.5,0.5")
            self.add_param(self.signal_params_layout, "env_shape", "gauss,gauss")
            self.add_param(self.signal_params_layout, "mod_type", "1,2")

    def add_param(self, layout, label, default_value):
        spinbox = QLineEdit(self)
        spinbox.setText(str(default_value))
        layout.addRow(label + ":", spinbox)

    def update_noise_params(self):
        for i in reversed(range(self.noise_params_layout.count())):
            item = self.noise_params_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget and widget not in [self.noise_type_combo, self.noise_level_spinbox]:
                    self.noise_params_layout.removeWidget(widget)
                    widget.deleteLater()
    
        noise_type = self.noise_type_combo.currentText()
        if noise_type == "Pseudo Noise":
            self.add_param(self.noise_params_layout, "pack_num", 3)
            self.add_param(self.noise_params_layout, "pack_dist", 1)
            self.add_param(self.noise_params_layout, "f_carr", 50)
            self.add_param(self.noise_params_layout, "tau", 5)
            self.add_param(self.noise_params_layout, "freq", "0,0.5")
        elif noise_type == "White Noise":
            self.add_param(self.noise_params_layout, "t_sig", 5)
            self.add_param(self.noise_params_layout, "sigma", 0.1)

    def toggle_noise_params(self):
        self.noise_params_layout.setEnabled(self.noise_checkbox.isChecked())
        if self.noise_checkbox.isChecked():
            self.update_noise_params()
        else:
            for i in reversed(range(self.noise_params_layout.count())):
                item = self.noise_params_layout.itemAt(i)
                if item:
                    widget = item.widget()
                    if widget and widget not in [self.noise_type_combo, self.noise_level_spinbox]:
                        self.noise_params_layout.removeWidget(widget)
                        widget.deleteLater()

    def generate_signal(self):
        signal_type = self.signal_type_combo.currentText()
        sampling_rate = self.sampling_rate_spinbox.value()
        signal_generator = SignalGenerator(sampling_rate)
        noise_generator = NoiseGenerator(sampling_rate)

        signal_params = self.get_signal_params(signal_type)
        noise_params = self.get_noise_params()

        if signal_type == "DF Signal":
            self.signal, self.time = signal_generator.df_signal(*signal_params)
        elif signal_type == "Tonal Signal":
            self.signal, self.time = signal_generator.tonal_signal(*signal_params)
        elif signal_type == "PFM Signal":
            self.signal, self.time = signal_generator.pfm_signal(*signal_params)
        elif signal_type == "LFM Signal":
            self.signal, self.time = signal_generator.lfm_signal(*signal_params)
        elif signal_type == "HFM Signal":
            self.signal, self.time = signal_generator.hfm_signal(*signal_params)
        elif signal_type == "Freq Domain Signal":
            self.signal, self.time, self.freq, self.S = freq_domain_signal_gen(*signal_params)

        if self.noise_checkbox.isChecked():
            noise_type = self.noise_type_combo.currentText()
            if noise_type == "Pseudo Noise":
                noise, _ = noise_generator.pseudo_noise(*noise_params)
            elif noise_type == "White Noise":
                noise, _ = noise_generator.white_noise(*noise_params[:2])

            # Приведение к одинаковой длине
            min_length = min(len(self.signal), len(noise))
            self.signal = self.signal[:min_length]
            noise = noise[:min_length]
            self.time = self.time[:min_length]

            self.signal += noise

        self.plot_signal(self.signal, sampling_rate, "Generated Signal")

    def get_signal_params(self, signal_type):
        params = []
        for i in range(self.signal_params_layout.rowCount()):
            item = self.signal_params_layout.itemAt(i, QFormLayout.FieldRole)
            if item:
                widget = item.widget()
                value = widget.text()
                if ',' in value:
                    try:
                        params.append([float(v) if v.replace('.', '', 1).isdigit() else v for v in value.split(',')])
                    except ValueError:
                        params.append(value.split(','))
                else:
                    try:
                        params.append(float(value))
                    except ValueError:
                        params.append(value)
        return params


    def get_noise_params(self):
        params = []
        for i in range(self.noise_params_layout.rowCount()):
            item = self.noise_params_layout.itemAt(i, QFormLayout.FieldRole)
            if item:
                widget = item.widget()
                if isinstance(widget, QComboBox):
                    value = widget.currentText()
                else:
                    value = widget.text()
                if ',' in value:
                    try:
                        params.append([float(v) if v.replace('.', '', 1).isdigit() else v for v in value.split(',')])
                    except ValueError:
                        params.append(value.split(','))
                else:
                    try:
                        params.append(float(value))
                    except ValueError:
                        params.append(value)
        return params

    def calculate_rays(self):
        if hasattr(self, 'signal'):
            sampling_rate = self.sampling_rate_spinbox.value()
            depth = self.depth_spinbox.value()
            temperature = self.temperature_spinbox.value()
            salinity = self.salinity_spinbox.value()
            speed_of_sound = self.calculate_speed_of_sound(depth, temperature, salinity)
            num_rays = 5  # количество лучей для моделирования
            max_distance = 500  # максимальное расстояние для расчета лучей (м)

            time_ray_signals = time_domain_calc_rays(self.signal, sampling_rate, speed_of_sound, num_rays, max_distance)
            self.plot_ray_signals(time_ray_signals, sampling_rate, "Time Domain Ray Signals")

            freq_ray_signals = freq_domain_calc_rays(self.signal, sampling_rate, speed_of_sound, num_rays, max_distance)
            self.plot_ray_signals(freq_ray_signals, sampling_rate, "Frequency Domain Ray Signals")


    def noise_detection_mode(self):
        if hasattr(self, 'signal'):
            sampling_rate = self.sampling_rate_spinbox.value()
            noise_threshold = self.noise_threshold_spinbox.value()  # порог шума, задаваемый пользователем
            
            # Пример обработки для режима шумопеленгования
            noise_indices = np.where(self.signal > noise_threshold)[0]
            noise_times = noise_indices / sampling_rate
            
            self.canvas.axes.clear()
            self.canvas.axes.plot(self.time, self.signal)
            self.canvas.axes.scatter(self.time[noise_indices], self.signal[noise_indices], color='red')
            self.canvas.axes.set_title("Noise Detection Mode")
            self.canvas.axes.set_xlabel("Time [s]")
            self.canvas.axes.set_ylabel("Amplitude")
            self.canvas.draw()

    def hydroacoustic_mode(self):
        if hasattr(self, 'signal'):
            sampling_rate = self.sampling_rate_spinbox.value()
            depth = self.depth_spinbox.value()
            temperature = self.temperature_spinbox.value()
            salinity = self.salinity_spinbox.value()
            speed_of_sound = self.calculate_speed_of_sound(depth, temperature, salinity)
            num_rays = 5  # количество лучей для моделирования
            max_distance = 500  # максимальное расстояние для расчета лучей (м)

            # Пример обработки для гидролокационного режима
            time_ray_signals = time_domain_calc_rays(self.signal, sampling_rate, speed_of_sound, num_rays, max_distance)
            self.plot_ray_signals(time_ray_signals, sampling_rate, "Hydroacoustic Mode - Time Domain Rays")


    def plot_signal(self, signal, sampling_rate, title):
        time = np.linspace(0, len(signal) / sampling_rate, num=len(signal))
        self.canvas.axes.clear()
        self.canvas.axes.plot(time, signal)
        self.canvas.axes.set_title(title)
        self.canvas.axes.set_xlabel("Time [s]")
        self.canvas.axes.set_ylabel("Amplitude")
        self.canvas.draw()


    def plot_ray_signals(self, ray_signals, sampling_rate, title):
        time = np.linspace(0, len(ray_signals[0]) / sampling_rate, num=len(ray_signals[0]))
        self.canvas.axes.clear()
        for i, ray_signal in enumerate(ray_signals):
            self.canvas.axes.plot(time, ray_signal, label=f'Ray {i+1}')
        self.canvas.axes.set_title(title)
        self.canvas.axes.set_xlabel("Time [s]")
        self.canvas.axes.set_ylabel("Amplitude")
        self.canvas.axes.legend()
        self.canvas.draw()


    def calculate_speed_of_sound(self, depth, temperature, salinity):
        # Формула Медисона и Маккензи для скорости звука в морской воде
        speed_of_sound = (1449.2 + 4.6 * temperature - 0.055 * temperature**2 + 0.00029 * temperature**3
                          + (1.34 - 0.01 * temperature) * (salinity - 35)
                          + 0.016 * depth)
        return speed_of_sound

def main():
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
