import sys
import os
import gc
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QFileDialog, QLabel, QProgressBar, QTextEdit, QHBoxLayout, QSlider)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from moviepy import VideoFileClip
sys.path.append("../core")

from video_analyzer import (extract_audio, analyze_audio_loudness, 
                                detect_highlights_gaming, make_highlight_with_subs, calculate_auto_threshold, get_visual_activity_score, make_highlight)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=2, dpi=100):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.tight_layout()
        super().__init__(fig)

class AnalysisWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list, object, float) 

    def __init__(self, file_path, threshold):
        super().__init__()
        self.file_path = file_path
        self.threshold = threshold

    def run(self):
        try:
            self.progress.emit("Inicjalizacja MoviePy...")
            with VideoFileClip(self.file_path) as clip_info:
                duration = clip_info.duration
            
            self.progress.emit("Ekstrakcja audio do analizy...")
            temp_audio = "temp_analysis.wav"
            extract_audio(self.file_path, temp_audio)

            self.progress.emit("Analiza głośności (Librosa/Pydub)...")
            loudness = analyze_audio_loudness(temp_audio)

            self.progress.emit("Wykrywanie wstępnych highlightów (Audio)...")
            initial_highlights = detect_highlights_gaming(loudness, duration, threshold=self.threshold)

            final_highlights = []
            if initial_highlights:
                self.progress.emit(f"Weryfikacja ruchu dla {len(initial_highlights)} momentów...")
                for start, end in initial_highlights:
                    v_score = get_visual_activity_score(self.file_path, start, end)
                    
                    if v_score > 1:
                        final_highlights.append((start, end))
                        self.progress.emit(f"Zaakceptowano klip: ruch {v_score:.2f}")
                    else:
                        self.progress.emit(f"Odrzucono klip (brak ruchu): {v_score:.2f}")
            
            highlights = final_highlights
            self.progress.emit("Obliczanie optymalnego progu (Auto-Threshold)...")
            auto_val = calculate_auto_threshold(loudness)

            gc.collect()
            if os.path.exists(temp_audio):
                try: os.remove(temp_audio)
                except: pass

            if not highlights:
                self.progress.emit("Brak fragmentów spełniających oba kryteria (Audio + Ruch).")
                self.finished.emit([], loudness, auto_val)
                return

            for i, (start, end) in enumerate(highlights):
                filename = f'highlight_{i+1}.mp4'
                self.progress.emit(f"Renderowanie {i+1}/{len(highlights)}...")
                make_highlight_with_subs(self.file_path, start, end, filename)
            
            self.finished.emit(highlights, loudness, auto_val)

        except Exception as e:
            self.progress.emit(f"BŁĄD: {str(e)}")
            print(f"Szczegóły błędu: {e}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Smart Video Highlighter - Gaming Edition")
        self.setMinimumSize(800, 600)
        

        self.last_loudness_data = None

        main_layout = QVBoxLayout()

        self.threshold_label = QLabel("Czułość detekcji: 80%")
        main_layout.addWidget(self.threshold_label)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)  # Zakres 0-100%
        self.threshold_slider.setValue(80)      # Domyślnie 0.8
        self.threshold_slider.valueChanged.connect(self.on_slider_move)
        main_layout.addWidget(self.threshold_slider)

        top_layout = QHBoxLayout()
        self.label = QLabel("Wybierz stream (.mp4)")
        top_layout.addWidget(self.label)
        
        self.import_button = QPushButton("Importuj Wideo")
        self.import_button.clicked.connect(self.open_file_dialog)
        top_layout.addWidget(self.import_button)
        main_layout.addLayout(top_layout)

        # Wykres
        self.canvas = MplCanvas(self)
        main_layout.addWidget(self.canvas)

        # Logi i status
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")
        main_layout.addWidget(self.log_output)

        # Pasek postępu
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Przycisk startu
        self.analyze_button = QPushButton("ROZPOCZNIJ ANALIZĘ")
        self.analyze_button.setFixedHeight(50)
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.start_analysis)
        main_layout.addWidget(self.analyze_button)

        self.setLayout(main_layout)


    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(filter="Wideo (*.mp4 *.mkv *.avi)")
        if file_path:
            self.file_path = file_path
            self.label.setText(f"Plik: {os.path.basename(file_path)}")
            self.analyze_button.setEnabled(True)
            self.log_output.append(f"Załadowano: {file_path}")


    def on_slider_move(self, value):
        self.threshold_label.setText(f"Czułość detekcji: {value}%")
        if self.last_loudness_data is not None:
            self.plot_data(self.last_loudness_data)
    
    def start_analysis(self):
        self.analyze_button.setEnabled(False)
        val = self.threshold_slider.value() / 100.0
        self.worker = AnalysisWorker(self.file_path, threshold=val)
        self.worker.progress.connect(self.update_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def update_log(self, message):
        self.log_output.append(message)

    def on_finished(self, highlights, loudness_data, auto_threshold):
        self.last_loudness_data = loudness_data
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.analyze_button.setEnabled(True)
        

        auto_percent = int(auto_threshold * 100)
        self.threshold_slider.setValue(auto_percent)
        self.threshold_label.setText(f"Czułość detekcji (Auto): {auto_percent}%")
        self.plot_data(loudness_data)
        
        if highlights:
            self.log_output.append(f"\nSUKCES! Wygenerowano {len(highlights)} plików.")
        else:
            self.log_output.append("\nAnaliza zakończona - nie znaleziono highlightów.")
        self.plot_data(loudness_data)

    def plot_data(self, data):
        self.canvas.axes.clear()
        current_threshold = self.threshold_slider.value() / 100.0
        
        self.canvas.axes.set_title("Znormalizowana Analiza Natężenia")
        self.canvas.axes.plot(data, color='#1f77b4', label='Głośność')
        self.canvas.axes.axhline(y=current_threshold, color='r', linestyle='--', label='Próg')
        
        self.canvas.axes.set_ylim(0, 1.1)
        self.canvas.axes.legend()
        self.canvas.draw()


    def update_threshold_label(self, value):
        self.threshold_label.setText(f"Próg czułości ({value})")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())