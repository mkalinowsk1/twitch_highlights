import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QFileDialog, QLabel, QProgressBar, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal
from core.video_analyzer import extract_audio, get_audio_energy, analyze_audio_loudness, normalize_and_combine, detect_highlights_gaming, make_highlight
from moviepy import VideoFileClip



# Klasa Worker obsługuje ciężkie obliczenia w tle
class AnalysisWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            # 1. Pobieranie metadanych
            self.progress.emit("Inicjalizacja...")
            with VideoFileClip(self.file_path) as clip_info:
                duration = clip_info.duration
            # Po wyjściu z 'with' clip_info jest na pewno zamknięte

            # 2. Ekstrakcja Audio
            self.progress.emit("Ekstrakcja audio...")
            temp_audio = "temp_audio_analysis.wav"
            # Używamy context managera wewnątrz extract_audio (poprawione wcześniej)
            extract_audio(self.file_path, temp_audio)

            # 3. Analiza (Ważne: upewnij się, że funkcje te nie blokują pliku)
            self.progress.emit("Analiza głośności...")
            loudness = analyze_audio_loudness(temp_audio)
            
            # USUWANIE PLIKU ANALIZY PRZED GENEROWANIEM KLIPÓW
            # To kluczowe: MoviePy może gryźć się z plikiem .wav w tym samym folderze
            try:
                import gc
                gc.collect() # Wymuszenie garbage collectora, by zwolnić librosę
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
            except:
                self.progress.emit("Ostrzeżenie: Nie udało się usunąć tymczasowego audio analizy.")

            self.progress.emit("Wykrywanie highlightów...")
            highlights = detect_highlights_gaming(loudness, duration)

            if not highlights:
                self.progress.emit("Nie znaleziono momentów.")
                self.finished.emit([])
                return

            # 4. Generowanie klipów
            for i, (start, end) in enumerate(highlights):
                output_file = f'highlight_{i+1}.mp4'
                self.progress.emit(f"Generowanie klipu {i+1}/{len(highlights)}...")
                # Przekazujemy absolutną ścieżkę, by uniknąć problemów z folderami
                abs_path = os.path.abspath(self.file_path)
                make_highlight(abs_path, start, end, output_file)
            
            self.finished.emit(highlights)

        except Exception as e:
            self.progress.emit(f"BŁĄD: {str(e)}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Smart Video Highlighter")
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout()

        self.label = QLabel("Wybierz plik wideo, aby rozpocząć")
        layout.addWidget(self.label)

        self.import_button = QPushButton("Importuj klip (.mp4)")
        self.import_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.import_button)

        self.analyze_button = QPushButton("Analizuj i generuj Highlighty")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)
        layout.addWidget(self.analyze_button)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(filter="MP4 Files (*.mp4)")
        if file_path:
            self.file_path = file_path
            self.label.setText(f"Wybrany plik: {os.path.basename(file_path)}")
            self.analyze_button.setEnabled(True)
            self.log_output.append(f"Załadowano: {file_path}")

    def start_analysis(self):
        self.analyze_button.setEnabled(False)
        self.progress_bar.setRange(0, 0) # Tryb "busy"
        self.worker = AnalysisWorker(self.file_path)
        self.worker.progress.connect(self.update_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def update_log(self, message):
        self.log_output.append(message)

    def on_finished(self, highlights):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.log_output.append(f"Zakończono! Znaleziono {len(highlights)} highlightów.")
        self.analyze_button.setEnabled(True)

app = QApplication(sys.argv)
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())