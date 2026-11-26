from PySide6.QtWidgets import  QApplication, QHBoxLayout, QLayout, QPushButton, QStyle, QTableWidget, QTableWidgetItem, QTextBrowser, QWidget
from PySide6.QtCore import Qt
from typing import Callable, Dict, List, Optional

class ViewManager:
    @staticmethod
    def add_widget_to_grid(
            layout: Optional[QLayout], 
            widget: QWidget, 
            index: int, 
            total_count: int
        ) -> None:

        """Dynamically distributes 1 to 4+ widgets in a responsive grid layout."""
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

    @staticmethod
    def get_active_layout(ui: QWidget, pages: Dict[str, int])-> Dict[str, Optional[QLayout]]:
        """Returns the currently active layout based on the current page index."""
        curr_idx = ui.stackedWidget.currentIndex()
        
        if curr_idx == pages["page_video_analysis"]:
            return {"video": ui.grid_layout_video, "audio": None}
        elif curr_idx == pages["page_audio_analysis"]:
            return {"video": None, "audio": ui.grid_layout_audio}
        elif curr_idx == pages["page_fusion_analysis"]:
            return {"video": ui.grid_video_details, "audio": ui.grid_audio_details}
        return {"video": None, "audio": None}
    @staticmethod
    def clear_layout(layout: Optional[QLayout]) -> None:
        """Clears all widgets from the given layout."""
        if layout is None:
            return
            
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
                continue
            #if the item is a layout, recursively clear it
            nested_layout = item.layout()
            if nested_layout is not None:
                ViewManager.clear_layout(nested_layout)

    @staticmethod
    def fill_incidents_table(
            table_widget: QTableWidget,
            incidents: List[dict],
            on_play: Callable[[str], None], #callback for play button
            on_status_change: Callable[[int, str], None], #callback for status change buttons
        ) -> None:
        """Fills the given QTableWidget with incident data."""
        table_widget.setRowCount(0)
        table_widget.setRowCount(len(incidents))
        table_widget.setColumnCount(6)

        style = QApplication.style()

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
            icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            btn_play.setIcon(icon)
            btn_play.setToolTip("Відтворити інцидент")
            btn_play.clicked.connect(lambda checked=False, file_path=inc['source_name']: on_play(file_path))

            btn_false = QPushButton()
            icon = style.standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton)
            btn_false.setIcon(icon)
            btn_false.setToolTip("Позначити інцидент як помилковий")
            btn_false.clicked.connect(lambda checked=False, id_inc=inc['id_incident']: on_status_change(id_inc, "Хибне тривога"))

            btn_resolve = QPushButton()
            icon = style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
            btn_resolve.setIcon(icon)
            btn_resolve.setToolTip("Позначити інцидент як вирішений")
            btn_resolve.clicked.connect(lambda checked=False, id_inc=inc['id_incident']: on_status_change(id_inc, "Вирішено"))

            layout.addWidget(btn_play)
            layout.addWidget(btn_false)
            layout.addWidget(btn_resolve)

            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            empty_item = QTableWidgetItem("")
            table_widget.setItem(row_idx, 5, empty_item)
            table_widget.setCellWidget(row_idx, 5, container)

    @staticmethod
    def render_fusion_tables(
        text_browser_audio: Optional[QTextBrowser], # QTextBrowser for audio fusion results
        text_browser_video: Optional[QTextBrowser], # QTextBrowser for video fusion results
        fusion_buffer: Dict[str, dict],             # Buffer containing fusion results for audio and video
    ) -> None:
        """Renders the fusion results in HTML tables for both audio and video QTextBrowsers."""
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

        def generate_rows(module_data: List[dict], module_type: str) -> str: 
            """Generates HTML rows for the given module data table (audio or video)."""
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

        for _, mods in fusion_buffer.items():
            audio_rows += generate_rows(mods.get("audio", []), "audio")
            video_rows += generate_rows(mods.get("video", []), "video")

        if text_browser_audio is not None:
            scroll_a = text_browser_audio.verticalScrollBar()
            pos_a = scroll_a.value()
            text_browser_audio.setHtml(table_start + audio_rows + table_end)
            scroll_a.setValue(pos_a)

        if text_browser_video is not None:
            scroll_v = text_browser_video.verticalScrollBar()
            pos_v = scroll_v.value()
            text_browser_video.setHtml(table_start + video_rows + table_end)
            scroll_v.setValue(pos_v)

    @staticmethod
    def update_aggression_banner(
        ui: QWidget,            
        fusion_conf: float,
        is_active: bool,
        source_name: str = "",
    ) -> None:
        """Updates the aggression banner based on the fusion confidence and source name."""
        widget_alert = getattr(ui, "widget_agg_alert", None)
        if widget_alert is None:
            return
        if is_active:
            ui.lbl_agg_prefix.setText(f"УВАГА! АГРЕСИВНА ПОВЕДІНКА [{source_name}]")
            ui.lbl_agg_confidence.setText(f"{fusion_conf:.1f}")
            ui.lbl_agg_suffix.setText("% впевненості)")
            widget_alert.show()
        else:
            widget_alert.hide()