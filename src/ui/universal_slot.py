from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QSizePolicy, QPushButton
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QPixmap
import os


from src.modules.video import video_train
from src.config import FONT_SIZE_SMALL, AUDIO_THRESHOLD
from src.modules.audio.audio_visualization import MplCanvas 

class UniversalSlot(QFrame):
    def __init__(self, source_path, id_source, has_video=False, has_audio=False, user_role="Аналітик"):
        super().__init__()
        self.source_path = source_path
        self.id_source = id_source
        self.name = os.path.basename(self.source_path)
        self.user_role = user_role
        self.current_segment_idx = -1

        self.audio_threshold = AUDIO_THRESHOLD
        self.video_threshold = video_train.load_threshold()

        self.audio_segments = []
        self.segment_predictions = []
        self.confidence_predictions = []
        
        self.setObjectName("UniversalSlot")

        self.has_audio = has_audio
        self.has_video = has_video
        self.setup_ui()
        
        if self.has_audio:
            self.audio_player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.audio_player.setAudioOutput(self.audio_output)
            
            file_url = QUrl.fromLocalFile(self.source_path)
            self.audio_player.setSource(file_url)
            self.audio_output.setMuted(True)
            
            self.canvas.init_plot()  

            self.timer = QTimer()
            self.timer.timeout.connect(self.on_timer_tick)
            self.timer.start(100) 

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.setFrameShape(QFrame.StyledPanel)

        self.lbl_title = QLabel(self.name)
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet(f"font-weight: bold; background-color: #333; font-size: {FONT_SIZE_SMALL}px; color: white; border: none;")
        self.main_layout.addWidget(self.lbl_title)

        
        if self.has_video: 
            self.video_screen = QLabel("ВІДЕО ПОТІК")
            self.video_screen.setMinimumSize(160, 90) 
            self.video_screen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.video_screen.setAlignment(Qt.AlignCenter)
            self.video_screen.setStyleSheet("background-color: white; border: 1px solid #333;")
            self.main_layout.addWidget(self.video_screen)

        self.canvas = MplCanvas()
        self.canvas.user_role = self.user_role
        if not self.has_audio: self.canvas.hide()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.canvas)

        
        self.lbl_status = QLabel("ОЧІКУВАННЯ АНАЛІЗУ")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("background-color: #333; border-radius: 5px; padding: 8px; font-weight: bold; border: none; color: white")
        self.main_layout.addWidget(self.lbl_status)

        if self.has_audio: 
            self.btn_mute = QPushButton("🔇 Увімкнути звук")
            self.btn_mute.setCursor(Qt.PointingHandCursor)
            self.btn_mute.setStyleSheet("""
                QPushButton { background-color: #444; color: white; border-radius: 5px; padding: 5px; border: none;}
                QPushButton:hover { background-color: #555; }
            """)
            self.btn_mute.clicked.connect(self.toggle_mute)
            self.main_layout.addWidget(self.btn_mute)
            
    def on_timer_tick(self):
        if self.audio_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            return

        current_time = self.audio_player.position() / 1000.0
        seg_idx = int(current_time / 3.0)
        
        if seg_idx >= len(self.segment_predictions):
            return 

        chunk = self.audio_segments[seg_idx] if seg_idx < len(self.audio_segments) else None
        prediction = float(self.segment_predictions[seg_idx])
        
        self.canvas.sync_ui(current_time, chunk, prediction)

        if seg_idx != self.current_segment_idx:
            self.current_segment_idx = seg_idx
            self.update_model_pred("audio", prediction)

    def update_video_frame(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        
        self.video_screen.setPixmap(pixmap.scaled(
            self.video_screen.width(), 
            self.video_screen.height(), 
            Qt.IgnoreAspectRatio, 
            Qt.SmoothTransformation
        ))
    def update_model_pred(self, pred_type, loss):
        self.last_pred = loss
        
        threshold = self.audio_threshold if pred_type == "audio" else self.video_threshold
        
        if pred_type == "audio":
            prefix = f"СЕГМЕНТ {self.current_segment_idx + 1}" if self.current_segment_idx >= 0 else "АУДІО"
        else:
            prefix = "ВІДЕО"

        is_alert = loss > threshold
        
        is_fusion = getattr(self, 'is_fusion_managed', False)

        if hasattr(self, 'lbl_status'):
            if is_alert:
                status_text = "КРИТИЧНО" if is_fusion else "АГРЕСІЯ"
                
                self.lbl_status.setText(f"{prefix}: {status_text} (Loss: {loss:.2f})")
                self.lbl_status.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold; padding: 8px; border-radius: 5px;")
            else:
                self.lbl_status.setText(f"{prefix}: НОРМА (Loss: {loss:.2f})")
                self.lbl_status.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 8px; border-radius: 5px;")
    
    def add_audio_segment(self, idx, pred, chunk):
        while len(self.audio_segments) <= idx:
            self.audio_segments.append(None)
            self.segment_predictions.append(-1.0)
            
        self.audio_segments[idx] = chunk
        self.segment_predictions[idx] = pred

    def toggle_mute(self):
        if not hasattr(self, 'audio_output'):
            return
            
        is_muted = self.audio_output.isMuted()
        self.audio_output.setMuted(not is_muted)
        
        if is_muted: 
            self.btn_mute.setText("🔊 Вимкнути звук")
            self.btn_mute.setStyleSheet("""
                QPushButton { background-color: #0c2461; color: white; border-radius: 5px; padding: 5px; border: none;}
                QPushButton:hover { background-color: #1e3799; }
            """)
        else: 
            self.btn_mute.setText("🔇 Увімкнути звук")
            self.btn_mute.setStyleSheet("""
                QPushButton { background-color: #0c2461; color: white; border-radius: 5px; padding: 5px; border: none;}
                QPushButton:hover { background-color: #1e3799; }
            """)
