import sys
import os
import numpy as np
import time
import datetime
import platform
import subprocess
import requests
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLineEdit, QMainWindow, QMessageBox, QFileDialog, QPushButton, QStyle, QTableWidgetItem, QInputDialog, QWidget
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QTimer, Slot
from PySide6.QtGui import QIcon, QPixmap, QPainter, QFont

from src.config import VIDEO_MODEL_PATH, AUDIO_MODEL_PATH, VIDEO_MODEL_NEW_PATH, AUDIO_MODEL_NEW_PATH, OUTPUTS_DIR, UI_PATH, SECRET_PIN,AUDIO_THRESHOLD, WEBHOOK_URL

try:
    from src.modules.video import video_train
    from src.modules.video.video_preprocess import video_preprocess
    from src.modules.audio import audio_train, audio_preprocess
    from src.database.db_manager import DatabaseManager
    from src.utils.dataset_utils import get_total_count, get_saved_counts, save_counts
    from src.core.media_manager import SmartMediaManager
    from src.ui.universal_slot import UniversalSlot
    from src.core.fusion import MultimodalFusionMLP
except ImportError as e:
    print(f"Помилка імпорту: {e}")
    video_train = video_preprocess = audio_train = audio_preprocess = None


ROLE_MESSAGES = {
    "auth_success": {
        "Оператор": "👋 Вітаємо, операторе! Систему активовано. Гарної та спокійної зміни!",
        "Аналітик": "🔐 [AUTH] Успішний вхід. Роль: Аналітик. Доступ надано."
    },
    "db_connected": {
        None: "🟢 З'єднання з базою даних встановлено.",
        "Оператор": "🟢 Зв'язок із сервером встановлено. Моніторинг готовий.",
        "Аналітик": "📡 [DB] SQL Server (mcr.microsoft.com/azure-sql-edge) підключено. Порт: 1433."
    },
    "incident_found": {
        "Оператор": "⚠️ УВАГА! Виявлено підозрілу активність. Будь ласка, перегляньте сегмент.",
        "Аналітик": "🔴 [DETECTION] Тригер аномалії. Впевненість: {confidence}%. ID Інциденту: {id_incident}."
    },
    "mark_status": {
        "Оператор": "✅ Дякуємо за допомогу! Позначку '{status}' збережено.",
        "Аналітик": "🔄 [DB] Статус інциденту #{id_incident} змінено на {status}користувачем {id_user}."
    },
    "audit_saved": {
        "Оператор": "💾 Роботу завершено. Ви працювали над файлом '{source_name}' протягом {duration}.",
        "Аналітик": "💾 [AUDIT] Сесію закрито (ID:{id_audit}). Файл: '{source_name}' | Тривалість: {duration}."
    },
    "source_file_loaded": {
        "Оператор": "📁 Файли'{name_file}' успішно завантажено. Система готова до аналізу.",
        "Аналітик": "📁 [SOURCE] Завантажено файл. Назва: {name_file}"
    },
    "source_camera_connected": {
        "Оператор": "🎥 Камери '{name_camera}' підключено. Розпочато пряму трансляцію.",
        "Аналітик": "🎥 [SOURCE] З'єднання з потоком встановлено. ID: {id_camera} | Джерело: {type} | Конфіг: {name}"
    }
}


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        ui_file = QFile(UI_PATH)
        if not ui_file.open(QFile.ReadOnly):
            print(f"ПОМИЛКА: Файл {UI_PATH} не знайдено!")
            sys.exit(-1)
        loader = QUiLoader()
        self.ui = loader.load(ui_file, self)
        ui_file.close()
        
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        font = QFont("Arial Emoji", 54) 
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🛡️")
        painter.end()

        app_icon = QIcon(pixmap)
        self.setWindowIcon(app_icon)
        QApplication.setWindowIcon(app_icon)
   
        self.is_training = False
        self.video_ready = False
        self.audio_ready = False
        self.current_video_model = VIDEO_MODEL_PATH
        self.current_audio_model = AUDIO_MODEL_PATH

        self.video_path = ""
        self.audio_path = ""
        self.video_thread = None
        
        self.canvas = None
        self.cursor_line = None
        self.segment_predictions = []
        self.confidence_predictions = []

        self.user_role = None
        self.id_user = None
        self.session_start_time = None

        self.db = DatabaseManager()
        self.id_current_source = {}
        self.id_current_incident = None

        self.last_global_notification = 0
        self.global_notification_cooldown = 45   
        self.last_alerted_id_incident = None

        self.manager = SmartMediaManager()
        self.slots = {}
        self.is_fusion_mode = False
        self.fusion_buffer = {}
        self.fusion_model = MultimodalFusionMLP()
        self.fusion_model.load_model()
        self.number_of_files = self.manager.number_of_processes
        

        self.pages = {
            "page_welcome": 0,
            "page_setup": 1,
            "page_video_analysis": 2,
            "page_audio_analysis": 3,
            "page_fusion_analysis": 4
        }
        self.ui.stackedWidget.setCurrentIndex(self.pages["page_welcome"])
        self.add_log("db_connected")

        
        self.ui.frame_roles_select.hide()

        connections = [
            {"parent": self.manager, "name": "error_signal",           "func": self.on_manager_error},
            {"parent": self.manager, "name": "video_frame_signal",     "func": self.on_video_frame_received},
            {"parent": self.manager, "name": "raw_pred_signal",       "func": self.on_raw_pred_received},
            {"parent": self.manager, "name": "fusion_decision_signal", "func": self.on_fusion_decision_received},
            {"parent": self.manager, "name": "audio_segment_ready",    "func": self.handle_audio_result},
            {"parent": self.manager, "name": "video_metadata_signal", "func": self.handle_video_analysis},

            {"name": "stackedWidget", "signal": "currentChanged", "func": self.on_page_changed},

            {"name": "btn_exit", "signal": "clicked", "func": self.close},
            {"name": "btn_login_operator", "signal": "clicked", "func": lambda: self.attempt_role_change("Оператор")},
            {"name": "btn_login_analyst", "signal": "clicked", "func": lambda: self.attempt_role_change("Аналітик")},

            {"name": "btn_welcome_start", "signal": "clicked", "func": self.on_welcome_clicked},

            {"name": "btn_back_setup", "signal": "clicked", "func": self.on_back_to_menu},
            {"name": "btn_back_video", "signal": "clicked", "func": self.on_back_to_menu},
            {"name": "btn_back_audio", "signal": "clicked", "func": self.on_back_to_menu},
            {"name": "btn_back_fusion", "signal": "clicked", "func": self.on_back_to_menu},
            {"name": "general_menu",           "signal": "triggered", "func": self.on_back_to_menu},

            {"name": "btn_load_file",  "signal": "clicked", "func": self.upload_file},
            {"name": "detection_video_action",  "signal": "triggered", "func": self.upload_file},
            {"name": "detection_audio_action",  "signal": "triggered", "func": self.upload_file},

            {"name": "btn_aud_seek_back",  "signal": "clicked", "func": lambda *args: self.seek(-3)},
            {"name": "btn_aud_seek_fwd",   "signal": "clicked", "func": lambda *args: self.seek(3)},
            {"name": "btn_aud_play_pause", "signal": "clicked", "func": self.toggle},
            {"name": "btn_vid_seek_back", "signal": "clicked", "func": lambda *args: self.seek(-3)},
            {"name": "btn_vid_seek_fwd",  "signal": "clicked", "func": lambda *args: self.seek(3)},
            {"name": "btn_vid_play_pause",  "signal": "clicked",   "func": self.toggle},

            {"name": "btn_goto_video",               "signal": "clicked",   "func": self.on_goto_video_details},
            {"name": "btn_goto_audio",               "signal": "clicked",   "func": self.on_goto_audio_details},
            
            {"name": "btn_train_video",              "signal": "clicked",   "func": lambda: self.run_training_pipeline('video')},
            {"name": "btn_train_audio",              "signal": "clicked",   "func": lambda: self.run_training_pipeline('audio')},
            {"name": "start_fit_video_model_action", "signal": "triggered", "func": lambda: self.run_training_pipeline('video')},
            {"name": "start_fit_audio_model_action", "signal": "triggered", "func": lambda: self.run_training_pipeline('audio')},

            {"name": "manual_cleaning_video_data_action", "signal": "triggered", "func": lambda: self.show_not_implemented_message()},
            {"name": "analytics_audio_model_action",      "signal": "triggered", "func": lambda: self.show_not_implemented_message()},
            {"name": "analytics_video_model_action",      "signal": "triggered", "func": lambda: self.show_not_implemented_message()},
            {"name": "btn_connect_cam",                   "signal": "clicked",   "func": lambda: self.show_not_implemented_message()},
        ]

        for item in connections:
            parent = item.get("parent", self.ui)
            target_obj = getattr(parent, item["name"])
            
            if "signal" in item:
                signal = getattr(target_obj, item["signal"])
            else:
                signal = target_obj
            
            signal.connect(item["func"])


        self.check_system_status()


    def attempt_role_change(self, target_role):
        if target_role == "Аналітик":
            
            password, ok = QInputDialog.getText(
                self, 
                "Авторизація Аналітика", 
                "Введіть майстер-пароль:", 
                QLineEdit.EchoMode.Password
            )
            
            if not ok or password != SECRET_PIN:
                QMessageBox.critical(self, "Помилка доступу", "Невірний пароль! Доступ заборонено.")
                self._apply_role_permissions("Оператор")
                return

        self._apply_role_permissions(target_role)
    def _apply_role_permissions(self, role):
        is_new_login = (self.user_role != role)
            
        self.user_role = role 
        
        if role == "Оператор":
            self.ui.start_fit_video_model_action.setVisible(False)
            self.ui.start_fit_audio_model_action.setVisible(False)
            self.ui.manual_cleaning_video_data_action.setVisible(False)
            self.ui.analytics_audio_model_action.setVisible(False)
            self.ui.analytics_video_model_action.setVisible(False)
            print("Режим Оператора: доступ обмежено.")

            self.id_user = 1
        else:
            self.ui.start_fit_video_model_action.setVisible(True)
            self.ui.start_fit_audio_model_action.setVisible(True)
            self.ui.manual_cleaning_video_data_action.setVisible(True)
            self.ui.analytics_audio_model_action.setVisible(True)
            self.ui.analytics_video_model_action.setVisible(True)
            print("Режим Аналітика: повний доступ.")   

            self.id_user = 2

        if is_new_login:
            self.add_log("auth_success")
            
        self.on_start_work_clicked()

    def add_log(self, event_key, **kwargs):
        message_template = ROLE_MESSAGES.get(event_key, {}).get(self.user_role, "Подія зафіксована.")

        try:
            final_text = message_template.format(**kwargs)
        except KeyError:
            final_text = message_template

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {final_text}"
       
        self.ui.textBrowser_video_logs.append(formatted_message)
        self.ui.textBrowser_video_logs.ensureCursorVisible()

        self.ui.textBrowser_audio_logs.append(formatted_message)
        self.ui.textBrowser_audio_logs.ensureCursorVisible()
    

    def load_incidents(self):
        incidents = self.db.get_confirmed_incidents()

        if hasattr(self.ui, 'table_video_incidents'):
            self._fill_incidents_table(self.ui.table_video_incidents, incidents)

        if hasattr(self.ui, 'table_audio_incidents'):
            self._fill_incidents_table(self.ui.table_audio_incidents, incidents)
    def _fill_incidents_table(self, table_widget, incidents):
        table_widget.setRowCount(0)
        table_widget.setRowCount(len(incidents))
        table_widget.setColumnCount(6)

        for row_idx, inc in enumerate(incidents):
            id_inc = QTableWidgetItem(str(inc['id_incident']))
            table_widget.setItem(row_idx, 0, id_inc)
            
            item_source = QTableWidgetItem(inc['source_name'])
            table_widget.setItem(row_idx, 1, item_source)
            
            item_time = QTableWidgetItem(inc['time_formatted'])
            table_widget.setItem(row_idx, 2, item_time)
            
            conf_str = f"{inc['confidence']:.1f}%" 
            item_conf = QTableWidgetItem(conf_str)
            table_widget.setItem(row_idx, 3, item_conf)
            
            item_operator = QTableWidgetItem(inc['user_name'])
            table_widget.setItem(row_idx, 4, item_operator)


            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(2, 2, 2, 2)
            layout.setSpacing(4)

            btn_play = QPushButton()
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            btn_play.setIcon(icon)
            btn_play.setToolTip("Відтворити інцидент")
            btn_play.clicked.connect(lambda checked=False, file_path=inc['source_name']: self.handle_play_incident(file_path))

            btn_false = QPushButton()
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton)
            btn_false.setIcon(icon)
            btn_false.setToolTip("Позначити інцидент як помилковий")
            btn_false.clicked.connect(lambda checked=False, id_inc=inc['id_incident']: self.handle_status_change(id_inc, "Хибне спрацювання"))

            btn_resolve = QPushButton()
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
            btn_resolve.setIcon(icon)
            btn_resolve.setToolTip("Позначити інцидент як вирішений")
            btn_resolve.clicked.connect(lambda checked=False, id_inc=inc['id_incident']: self.handle_status_change(id_inc, "Вирішено"))

            layout.addWidget(btn_play)
            layout.addWidget(btn_false)
            layout.addWidget(btn_resolve)

            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            empty_item = QTableWidgetItem("")
            table_widget.setItem(row_idx, 5, empty_item)
            table_widget.setCellWidget(row_idx, 5, container)


    def check_system_status(self):
        if self.is_training:
            return
        
        video_count = get_total_count('video')
        self.ui.lbl_video_count.setText(f"Навчальна вибірка: {video_count} файлів")
            
        if os.path.exists(VIDEO_MODEL_NEW_PATH):
            self.video_ready = True
            self.current_video_model = VIDEO_MODEL_NEW_PATH
        elif os.path.exists(VIDEO_MODEL_PATH):
            self.video_ready = True
            self.current_video_model = VIDEO_MODEL_PATH
        else:
            self.video_ready = False

        
        if self.video_ready:
            status_text = "Статус: Оновлена" if "new" in self.current_video_model else "Статус: Натренована"
            self.ui.lbl_video_status.setText(status_text)
            self.ui.lbl_video_status.setStyleSheet("color: black; border: none;")
            self.ui.btn_train_video.setText("Перетренувати") 
        else:
            self.ui.lbl_video_status.setText("Статус: Не натренована")
            self.ui.lbl_video_status.setStyleSheet("color: #c0392b; border: none;")
            self.ui.btn_train_video.setText("Навчити") 
        audio_count = get_total_count('audio')
        
        self.ui.lbl_audio_count.setText(f"Навчальна вибірка: {audio_count} файлів")
            
        if os.path.exists(AUDIO_MODEL_NEW_PATH):
            self.audio_ready = True
            self.current_audio_model = AUDIO_MODEL_NEW_PATH
        elif os.path.exists(AUDIO_MODEL_PATH):
            self.audio_ready = True
            self.current_audio_model = AUDIO_MODEL_PATH
        else:
            self.audio_ready = False

        if self.audio_ready:
            status_text = "Статус: Оновлена" if "new" in self.current_audio_model else "Статус: Натренована"
            self.ui.lbl_audio_status.setText(status_text)
            self.ui.lbl_audio_status.setStyleSheet("color: black; border: none;")
            self.ui.btn_train_audio.setText("Перетренувати") 
        else:
            self.ui.lbl_audio_status.setText("Статус: Не натренована")
            self.ui.lbl_audio_status.setStyleSheet("color: #c0392b; border: none;")
            self.ui.btn_train_audio.setText("Навчити") 
        
        if self.video_ready or self.audio_ready:
            self.ui.btn_load_file.setEnabled(True)
            self.ui.btn_load_file.setText("ОБРАТИ ФАЙЛ ДЛЯ АНАЛІЗУ")
        else:
            self.ui.btn_load_file.setEnabled(False)
            self.ui.btn_load_file.setText("СПОЧАТКУ НАТРЕНУЙТЕ МОДЕЛІ")
    def run_training_pipeline(self, mode):
        if mode == 'video':
            name_ukr = "Відео"
            current_files = get_total_count('video')
            target_model_path = VIDEO_MODEL_NEW_PATH
            status_label = self.ui.lbl_video_status
            train_module = video_train
            process_module = video_preprocess
            epochs_est = 5
            est_time_per_file = 0.0053
        else:
            name_ukr = "Аудіо"
            current_files = get_total_count('audio')
            target_model_path = AUDIO_MODEL_NEW_PATH
            status_label = self.ui.lbl_audio_status
            train_module = audio_train
            process_module = audio_preprocess
            epochs_est = 30
            est_time_per_file = 0.2

        folder_prefix = f"processed_{mode}_parts"

        if os.path.exists(OUTPUTS_DIR):
            target_folders = [
                d for d in os.listdir(OUTPUTS_DIR) 
                if d.startswith(folder_prefix) and os.path.isdir(os.path.join(OUTPUTS_DIR, d))
            ]
            if target_folders:
                for folder in target_folders:
                    folder_path = os.path.join(OUTPUTS_DIR, folder)
                    if any(os.path.isfile(os.path.join(folder_path, f)) for f in os.listdir(folder_path)):
                        data_exists = True
                        break

        saved_counts = get_saved_counts()
        last_files = saved_counts.get(mode, 0)

        est_min = int((current_files * est_time_per_file * epochs_est) / 60)
        est_max = int(est_min * 1.3) + 2
        time_msg = f"\n\nОрієнтовний час навчання: {est_min}-{est_max} хв."

        msg = QMessageBox(self)
        msg.setWindowTitle(f"Тренування {name_ukr}")
        msg.setIcon(QMessageBox.Question)

        needs_processing = False 

        if not data_exists:
            msg.setText(f"Файли оброблених даних {name_ukr} відсутні.")
            msg.setInformativeText("Необхідно запустити повну обробку перед навчанням.\n" + time_msg)
            btn_full = msg.addButton("Запустити обробку та навчання", QMessageBox.AcceptRole)
            btn_cancel = msg.addButton("Скасувати", QMessageBox.RejectRole)
            msg.exec()

            if msg.clickedButton() == btn_cancel:
                return
            needs_processing = True

        else:
            diff_text = ""
            if current_files != last_files:
                diff_text = f"УВАГА: Кількість файлів змінилась (Було: {last_files}, Зараз: {current_files}).\nРекомендовано повну обробку."
            else:
                diff_text = "Змін у кількості файлів не виявлено."

            msg.setText(f"Як ви хочете запустити навчання {name_ukr}?")
            msg.setInformativeText(f"{diff_text}\n{time_msg}\n\nОберіть дію:")

            btn_full = msg.addButton("Повна обробка + Навчання", QMessageBox.YesRole)
            btn_fast = msg.addButton("Тільки навчання (Старі дані)", QMessageBox.NoRole)
            btn_cancel = msg.addButton("Скасувати", QMessageBox.RejectRole)
            
            msg.exec()

            clicked = msg.clickedButton()

            if clicked == btn_cancel:
                return
            elif clicked == btn_full:
                needs_processing = True  
            elif clicked == btn_fast:
                needs_processing = False 

        try:
            status_label.setStyleSheet("color: #e67e22; font-weight: bold;")
            QApplication.processEvents()

            if needs_processing:
                status_label.setText("Статус: Обробка даних...")
                QApplication.processEvents()
                
                if process_module:
                    process_module.run() 
                else:
                    QMessageBox.information(self, "Помилка", "Модуль обробки не знайдено!")
            else:
                print(f"[{mode}] Пропуск обробки даних (вибір користувача).")
            
            status_label.setText("Статус: Тренування...")
            QApplication.processEvents()

            
            if train_module:
                train_module.run(model_file=target_model_path)
            else:
                raise ImportError(f"Модуль тренування {mode} не знайдено")

            if needs_processing:
                saved_counts[mode] = current_files
                save_counts(saved_counts)

            QMessageBox.information(self, "Готово", f"Модель {name_ukr} успішно збережено:\n{os.path.basename(target_model_path)}")
            self.is_training = False
            self.check_system_status()

        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Сталася помилка:\n{e}")
            self.check_system_status()


    def save_audit_for_source(self, id_source, start_time, source_path):
        end_time = datetime.datetime.now()
        
        delta = end_time - start_time
        duration_str = f"{delta.total_seconds():.2f} сек"

        id_audit = self.db.create_audit_record(
            id_user=self.id_user,
            id_source=id_source,
            start_time=start_time,
            end_time=end_time
        )

        self.add_log(
            "audit_saved", 
            id_audit=id_audit, 
            source_name=source_path, 
            duration=duration_str
        )
    def save_all_sessions(self):
        for _, components in self.slots.items():
            slot = components.get('video') or components.get('audio')
            
            if slot:
                self.save_audit_for_source(
                    id_source=slot.id_source, 
                    start_time=self.session_start_time, 
                    source_path=slot.source_path
                )
        
        self.id_current_source = {} 
        self.slots = {}
        self.session_start_time = None


    def upload_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Media Files (*.mp4 *.avi *.mov *.mkv *.mpeg* .wav *.mp3 *.m4a *.flac *.ogg)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            if files:
                self.add_log("source_file_loaded", name_file=files)
                self.process_files(files)
    def process_files(self, source_paths):
        self.manager.stop_all()
        self.save_all_sessions()
        self.fusion_buffer.clear()
        self.slots = {}
        self.id_current_source = {}
        self.session_start_time = datetime.datetime.now()

        files_to_process = source_paths[:self.number_of_files]
        any_video = any(self.manager.has_video(p) for p in files_to_process)
        any_audio = any(self.manager.has_audio(p) for p in files_to_process)
        self.is_fusion_mode = any_video and any_audio

        target_page_key = "page_fusion_analysis" if self.is_fusion_mode else ("page_audio_analysis" if any_audio else "page_video_analysis")
        self.ui.stackedWidget.setCurrentIndex(self.pages[target_page_key])
        
        layouts = self.get_active_layout() 
        if layouts["video"]: self.clear_layout(layouts["video"])
        if layouts["audio"]: self.clear_layout(layouts["audio"])

        total_videos = sum(1 for p in files_to_process if self.manager.has_video(p))
        total_audios = sum(1 for p in files_to_process if self.manager.has_audio(p))

        video_idx = 0
        audio_idx = 0
        
        for path in files_to_process:
            db_id = self.db.get_or_create_file_source(path)
            self.id_current_source[path] = db_id
            self.slots[path] = {} 
            if self.manager.has_video(path):
                slot_v = UniversalSlot(path, db_id, has_video=True, has_audio=False, user_role=self.user_role)
                slot_v.is_fusion_managed = self.is_fusion_mode
                self.slots[path]['video'] = slot_v
                self._add_widget_to_grid(layouts['video'], slot_v, video_idx, total_videos)
                video_idx += 1
                
            if self.manager.has_audio(path):
                slot_a = UniversalSlot(path, db_id, has_video=False, has_audio=True, user_role=self.user_role)
                slot_a.is_fusion_managed = self.is_fusion_mode
                self.slots[path]['audio'] = slot_a
                self._add_widget_to_grid(layouts['audio'], slot_a, audio_idx, total_audios)
                audio_idx += 1
                self.manager.register_audio_player(path, slot_a.audio_player)

        self.manager.ingest_files(files_to_process)

        for path, comps in self.slots.items():
            if 'audio' in comps and hasattr(comps['audio'], 'audio_player'):
                comps['audio'].audio_player.play()
    def _add_widget_to_grid(self, layout, widget, index, total_count):
        if total_count == 1:
            layout.addWidget(widget, 0, 0)
        elif total_count == 2:
            layout.addWidget(widget, index, 0)
        elif total_count == 3:
            if index == 0:
                layout.addWidget(widget, 0, 0, 1, 2) 
            elif index == 1:
                layout.addWidget(widget, 1, 0)
            else:
                layout.addWidget(widget, 1, 1)
        else: 
            row = index // 2
            col = index % 2
            layout.addWidget(widget, row, col)
    

    def get_active_layout(self):
        curr_idx = self.ui.stackedWidget.currentIndex()
        
        if curr_idx == self.pages["page_video_analysis"]:
            return {"video": self.ui.grid_layout_video, "audio": None}
        elif curr_idx == self.pages["page_audio_analysis"]:
            return {"video": None, "audio": self.ui.grid_layout_audio}
        elif curr_idx == self.pages["page_fusion_analysis"]:
            return {"video": self.ui.grid_video_details, "audio": self.ui.grid_audio_details}
        return None
    def clear_layout(self, layout):
        if layout is None:
            return
            
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


    def send_google_chat_alert(self,loss_value, source_name="Невідомо"):
        import threading
        
        webhook_url = WEBHOOK_URL
        message=(
            f"🚨 *ТРИВОГА: ВИЯВЛЕНО АГРЕСІЮ*\n"
            f"------------------------------------------\n"
            f"*Локація:* {source_name}\n"
            f"*Рівень аномалії:* `{loss_value:.4f}`\n"
            f"*Час виявлення:* {time.strftime('%H:%M:%S, %d.%m.%Y')}\n"
            f"------------------------------------------\n"
        )

        payload = {"text": message}
    
        def send_request():
            try:
                requests.post(webhook_url, json=payload, timeout=5)
            except Exception as e:
                print(f"Помилка відправки в Google Chat: {e}")

        threading.Thread(target=send_request, daemon=True).start()
        return True
    def trigger_incident_alert(self, media_type, confidence, time_sec, media_path):
        now = time.time()
        
        if not hasattr(self, 'source_cooldowns'):
            self.source_cooldowns = {}
            
        last_time = self.source_cooldowns.get(media_path, 0)
        if now - last_time < 10: 
            return
            
        self.source_cooldowns[media_path] = now

        status = "Підтверджено" if "Злиття" in media_type else "Критично"

        id_incident = self.db.process_incident_segment(
            id_source=self.id_current_source[media_path],
            id_user=self.id_user,
            media_type=media_type,
            current_time=int(time_sec),
            status=status,
            confidence=confidence,
            media_path=media_path
        )
        
        self.id_current_incident = id_incident
        self.add_log("incident_found", confidence=round(confidence, 1), id_incident=id_incident)

        self.load_incidents() 

        if now - self.last_global_notification > self.global_notification_cooldown:
            message = f'КРИТИЧНА ПОДІЯ 🚨!\nТип: {media_type}\nЧас: {int(time_sec)}с\nВпевненість: {round(confidence, 1)}%'
            
            original_volume = self.manager.get_global_volume()
            self.manager.set_global_volume(0.1)
            os.system("afplay /System/Library/Sounds/Sosumi.aiff &")
            QTimer.singleShot(3000, lambda: self.manager.set_global_volume(original_volume))

            script = f'display notification "{message}" with title "🛡️ ТВІЙ ВАРТОВИЙ"'
            QTimer.singleShot(200, lambda: os.system(f"osascript -e '{script}'"))

            self.send_google_chat_alert(confidence, source_name=os.path.basename(media_path))

            self.last_global_notification = now


    def receive_fusion_data(self, source_path, source_name, module_type, conf, time_sec, is_aggression, iou=0.0, num_people=1):
        if conf == -1: return
        try:
            if source_path not in self.fusion_buffer:
                self.fusion_buffer[source_path] = {"audio": [], "video": []}

            current_list = self.fusion_buffer[source_path][module_type]
            safe_time_end = max(0.0, time_sec)
            conf = max(0.0, min(100.0, conf))

            if module_type == "audio":
                safe_time_start = max(0.0, safe_time_end - 3.0)
            else:
                safe_time_start = safe_time_end

            def create_new_event():
                return {
                    "name": source_name,
                    "is_aggression": is_aggression,
                    "start_raw": safe_time_start,
                    "end_raw": safe_time_end,
                    "mean_conf": conf,
                    "count": 1,
                    "iou": iou,
                    "num_people": num_people
                }

            if len(current_list) == 0:
                current_list.append(create_new_event())
            else:
                last_event = current_list[-1]

                if last_event["is_aggression"] == is_aggression:
                    time_diff = safe_time_end - last_event["end_raw"]
                    
                    if time_diff <= 4.0:
                        last_event["end_raw"] = safe_time_end
                        old_mean = last_event["mean_conf"]
                        old_count = last_event["count"]
                        last_event["mean_conf"] = ((old_mean * old_count) + conf) / (old_count + 1)
                        last_event["count"] += 1
                        
                        last_event["iou"] = max(last_event.get("iou", 0), iou)
                        last_event["num_people"] = max(last_event.get("num_people", 1), num_people)
                    else:
                        current_list.append(create_new_event())
                else:
                    current_list.append(create_new_event())

            self.update_fusion_ui()
            self.check_fusion_conditions(source_path)
            
        except Exception as e:
            print(f"ПОМИЛКА в receive_fusion_data: {e}")
    def update_fusion_ui(self):
        try:
            table_start = """
            <table width="100%" cellspacing="0" cellpadding="8">
                <tr style="color: #7f8c8d; font-size: 10pt; background-color: #f8f9fa;">
                    <th width="30%" align="left">Джерело</th>
                    <th width="20%" align="center">Впевненість</th>
                    <th width="20%" align="center">Тривалість</th>
                    <th width="25%" align="center">Мітка</th>
                </tr>
            """
            table_end = "</table>"
            audio_rows, video_rows = "", ""

            def generate_rows(module_data, module_type):
                rows_html = ""
                if not module_data: return rows_html

                recent_events = module_data
                rowspan_count = len(recent_events) 

                for i, ev in enumerate(recent_events):
                    name = ev["name"]
                    if ev["is_aggression"]:
                        c_conf = "#c0392b"
                        l_text = "Критично" if module_type == "audio" else "Агресія"
                        c_label = "#e67e22" if module_type == "audio" else "#c0392b"
                    else:
                        c_conf, l_text, c_label = "#27ae60", "Норма", "#27ae60"

                    conf_str = f'<span style="color:{c_conf};">{ev["mean_conf"]:.1f}%</span>'

                    t_start = f"{int(ev['start_raw'] // 60):02d}:{int(ev['start_raw'] % 60):02d}"
                    t_end = f"{int(ev['end_raw'] // 60):02d}:{int(ev['end_raw'] % 60):02d}"

                    if t_start == t_end:
                        dur_str = t_start
                    else:
                        dur_str = f"{t_start}-{t_end}"

                    label_str = f'<span style="color:{c_label}; font-weight:bold;">{l_text}</span>'

                    if i == 0:
                        name_td = f'<td width="30%" rowspan="{rowspan_count}" align="left" style="border-bottom: 1px solid #bdc3c7; vertical-align: middle;"><b>{name}</b></td>'
                    else:
                        name_td = "" 

                    rows_html += f"""
                    <tr>
                        {name_td}
                        <td width="20%" align="center" style="border-bottom: 1px solid #ecf0f1;">{conf_str}</td>
                        <td width="20%" align="center" style="border-bottom: 1px solid #ecf0f1;">{dur_str}</td>
                        <td width="25%" align="center" style="border-bottom: 1px solid #ecf0f1;">{label_str}</td>
                    </tr>
                    """
                return rows_html

            for _, mods in self.fusion_buffer.items():
                audio_rows += generate_rows(mods.get("audio", []), "audio")
                video_rows += generate_rows(mods.get("video", []), "video")

            if hasattr(self.ui, 'textBrowser_audio'):
                tb_audio = self.ui.textBrowser_audio
                scroll_a = tb_audio.verticalScrollBar()
                pos_a = scroll_a.value()
                tb_audio.setHtml(table_start + audio_rows + table_end)
                scroll_a.setValue(pos_a)

            if hasattr(self.ui, 'textBrowser_video'):
                tb_video = self.ui.textBrowser_video
                scroll_v = tb_video.verticalScrollBar()
                pos_v = scroll_v.value()
                tb_video.setHtml(table_start + video_rows + table_end)
                scroll_v.setValue(pos_v)
                
        except Exception as e:
            print(f"ПОМИЛКА в update_fusion_ui: {e}")
    def check_fusion_conditions(self, source_path):
        try:
            data = self.fusion_buffer.get(source_path)
            if not data: 
                return
                
            audio_list = data.get("audio", [])
            video_list = data.get("video", [])

            for aud_event in audio_list:
                if not aud_event["is_aggression"]: 
                    continue
                
                for vid_event in video_list:
                    if not vid_event["is_aggression"]: 
                        continue
                    
                    # Логіка перетину інтервалів
                    if (aud_event["start_raw"] < vid_event["end_raw"] + 1.0) and \
                       (vid_event["start_raw"] < aud_event["end_raw"] + 1.0):
                        
                        audio_prob = aud_event["mean_conf"] / 100.0
                        video_prob = vid_event["mean_conf"] / 100.0
                        iou = vid_event.get("iou", 0.0)
                        num_people = vid_event.get("num_people", 1)
                        
                        prob, label = self.fusion_model.predict(audio_prob, video_prob, iou, num_people)
                        fusion_conf = prob * 100.0
                        
                        incident_start_time = max(1, min(aud_event["start_raw"], vid_event["start_raw"]))

                        if label == 1: 
                            if hasattr(self.ui, 'widget_agg_alert'):
                                self.ui.lbl_agg_prefix.setText(f"УВАГА! АГРЕСИВНА ПОВЕДІНКА [{aud_event['name']}]")
                                self.ui.lbl_agg_confidence.setText(f"{fusion_conf:.1f}")
                                self.ui.lbl_agg_suffix.setText("% впевненості)")
                                self.ui.widget_agg_alert.show()
                            
                            self.trigger_incident_alert(
                                media_type="Злиття",
                                confidence=fusion_conf,
                                time_sec=int(incident_start_time),
                                media_path=source_path
                            )
                            return

            any_active = any(e["is_aggression"] for e in audio_list + video_list)
            if not any_active and hasattr(self.ui, 'widget_agg_alert'):
                self.ui.widget_agg_alert.hide()

        except Exception as e:
            print(f"ПОМИЛКА в check_fusion_conditions: {e}")
    def switch_analysis_page(self, target_page_key):
        self.ui.stackedWidget.setCurrentIndex(self.pages[target_page_key])
        
        layouts = self.get_active_layout()
        
        total_videos = sum(1 for comps in self.slots.values() if 'video' in comps)
        total_audios = sum(1 for comps in self.slots.values() if 'audio' in comps)

        video_idx = 0
        audio_idx = 0
        
        for _, comps in self.slots.items():
            if 'video' in comps and layouts["video"] is not None:
                self._add_widget_to_grid(layouts['video'], comps['video'], video_idx, total_videos)
                video_idx += 1
                
            if 'audio' in comps and layouts["audio"] is not None:
                self._add_widget_to_grid(layouts['audio'], comps['audio'], audio_idx, total_audios)
                audio_idx += 1
    
    def on_start_work_clicked(self):
        if self.user_role == "Оператор":
            self.upload_file() 
        else:
            self.ui.stackedWidget.setCurrentIndex(self.pages["page_setup"])
    def on_welcome_clicked(self):
        self.ui.btn_welcome_start.hide()
        self.ui.frame_roles_select.show()
    def on_back_to_menu(self):
        current_idx = self.ui.stackedWidget.currentIndex()
        
        if self.is_fusion_mode and current_idx in (self.pages["page_video_analysis"], self.pages["page_audio_analysis"]):
            self.switch_analysis_page("page_fusion_analysis")
        elif self.user_role == "Аналітик":
            self.ui.stackedWidget.setCurrentIndex(self.pages["page_setup"])
        else:
            self.ui.stackedWidget.setCurrentIndex(self.pages["page_welcome"])
            
            self.save_all_sessions()
        self.load_incidents()
    def show_not_implemented_message(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("У розробці")
        msg.setText("Ця функція ще не реалізована.")
        msg.setInformativeText("Дякую за розуміння! 🥺\nМи працюємо над цим.")
        msg.exec()
    def close(self):
        self.save_all_sessions()
        self.db.stop_docker_container()
        QApplication.instance().quit()
    def on_goto_video_details(self):
        self.switch_analysis_page("page_video_analysis")
        self.load_incidents()
    def on_goto_audio_details(self):
        self.switch_analysis_page("page_audio_analysis")
        self.load_incidents()
    def on_page_changed(self, index):
        idle_pages = [self.pages["page_welcome"], self.pages["page_setup"]]
        
        if index in idle_pages:
            if self.ui.btn_aud_play_pause.text() == "⏸️":
                self.toggle()
            if self.ui.btn_vid_play_pause.text() == "⏸️":
                self.toggle()
                
            self.manager.set_global_volume(0.0)
        else:
            self.manager.set_global_volume(1.0)
    
    def handle_status_change(self, id_incident, new_status):
        current_role = getattr(self, 'user_role', 'Оператор') 
        
        if current_role != 'Аналітик':
            QMessageBox.warning(
                self, 
                "Обмеження доступу", 
                "Увага: Тільки користувач із роллю «Аналітик» має право закривати інциденти або позначати їх як хибні."
            )
            return 
        
        if new_status == "Вирішено":
            title = "Підтвердження вирішення"
            msg = f"Ви впевнені, що хочете закрити реальний інцидент #{id_incident} як 'Вирішено'?"
        else:
            title = "Підтвердження хибної тривоги"
            msg = f"Ви впевнені, що інцидент #{id_incident} є помилкою алгоритму (Хибне спрацювання)?"

        reply = QMessageBox.question(
            self, title, msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self.db.mark_status(id_incident, status=new_status)
                self.add_log("mark_status", id_incident=id_incident, id_user=self.id_user, status=new_status)
                QMessageBox.information(self, "Готово", f"Інцидент #{id_incident} успішно позначено як '{new_status}'.")
                self.load_incidents()
            except Exception as e:
                QMessageBox.critical(self, "Помилка", f"Не вдалося оновити статус:\n{e}")
        
        self.load_incidents()

    def handle_play_incident(self, file_path):
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(
                self, 
                "Файл не знайдено", 
                f"Не вдалося знайти медіафайл за вказаним шляхом:\n{file_path}"
            )
            return

        try:
            current_os = platform.system()
            if current_os == "Darwin":  
                subprocess.run(["open", file_path], check=True)
            elif current_os == "Windows":
                os.startfile(file_path)
            else:  # Linux
                subprocess.run(["xdg-open", file_path], check=True)
                
        except Exception as e:
            print(f"Помилка відкриття медіафайлу системним плеєром: {e}")
            QMessageBox.critical(
                self, 
                "Помилка відтворення", 
                f"Не вдалося відкрити файл стандартним плеєром ОС:\n{str(e)}"
            )

    def seek(self, seconds):
        self.manager.seek_all(seconds)
    def toggle(self):
        is_paused = self.manager.toggle_all()
        
        label = "▶️" if is_paused else "⏸️"
        
        self.ui.btn_aud_play_pause.setText(label)
        self.ui.btn_vid_play_pause.setText(label)


    @Slot(str, str)
    def on_manager_error(self, name: str, message: str) -> None:
        if name in self.slots:
            components = self.slots[name]
            
            if isinstance(components, dict):
                for slot_type, slot_obj in components.items():
                    if hasattr(slot_obj, 'lbl_status'):
                        slot_obj.lbl_status.setText(f"ПОМИЛКА: {message}")
                        slot_obj.lbl_status.setStyleSheet("background-color: #95a5a6; color: white;")
            
            else:
                components.lbl_status.setText(f"ПОМИЛКА: {message}")
    @Slot(str, str, float)
    def on_raw_pred_received(self, name: str, pred_type: str, loss: float) -> None:
        if name in self.slots and pred_type in self.slots[name]:
            self.slots[name][pred_type].update_model_pred(pred_type, loss)
    @Slot(str, float, bool)
    def on_fusion_decision_received(self, name: str, final_loss: float, status: bool) -> None:
        if self.ui.stackedWidget.currentIndex() != self.pages['page_fusion_analysis']:
            return
    @Slot(str, int, float, np.ndarray)
    def handle_audio_result(self, source_path: str, idx: int, pred: float, chunk: np.ndarray) -> None:
        if pred == -1:
            return
        if source_path in self.slots and 'audio' in self.slots[source_path]:
            slot = self.slots[source_path]['audio']
            slot.add_audio_segment(idx, pred, chunk)
            

            is_aggression = pred >= AUDIO_THRESHOLD
            display_conf = max(0.0, min(100.0, float(pred * 100.0)))
            current_time = float((idx + 1) * 3.0)
            
            if self.is_fusion_mode:
                self.receive_fusion_data(source_path, os.path.basename(source_path), "audio", display_conf, current_time, is_aggression)
            else:
                if is_aggression:
                    self.trigger_incident_alert(
                        media_type="Аудіо",
                        confidence=display_conf,
                        time_sec=int(current_time),
                        media_path=source_path
                    )

    @Slot(str, object)
    def on_video_frame_received(self, source_path:str, QImage_frame:object) -> None:
        if source_path in self.slots and 'video' in self.slots[source_path]:
            self.slots[source_path]['video'].update_video_frame(QImage_frame)
    @Slot(str, float, float, bool)
    @Slot(str, float, float, bool, float, int) 
    def handle_video_analysis(self, name: str, loss: float, current_time: float, is_bullying: bool, iou: float = 0.0, num_people: int = 1) -> None:
        if name not in self.slots:
            return
            
        threshold = video_train.load_threshold()
        max_logical_error = threshold * 4.0 
        clamped_conf = min(loss, max_logical_error)

        if clamped_conf >= threshold:
            display_conf = 50.0 + ((clamped_conf - threshold) / (max_logical_error - threshold)) * 50.0
        else:
            display_conf = (clamped_conf / threshold) * 50.0

        if self.is_fusion_mode:
            self.receive_fusion_data(name, os.path.basename(name), "video", display_conf, current_time, is_bullying, iou, num_people)
        else:
            if is_bullying:
                self.trigger_incident_alert(
                    media_type="Відео",
                    confidence=display_conf,
                    time_sec=int(current_time),
                    media_path=name
                )

if __name__ == "__main__":
    os.environ["QT_MAC_WANTS_LAYER"] = "0"
    app = QApplication(sys.argv)
    
    font = QFont("Helvetica Neue")
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    window = App()
    window.ui.showMaximized()
    
    sys.exit(app.exec())