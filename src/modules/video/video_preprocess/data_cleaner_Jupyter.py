import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.models import load_model
from src.modules.video.video_train import GraphConvolution

from src.config import PROCESSED_MANUALLY_CLEANED_VIDEO_PARTS_DIR as CLEANED_DIR


class DataCleanerJupyter:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        
        if not os.path.exists(data_path):
            return

        data = np.load(data_path, allow_pickle=True)
        
        self.datasets = {
            'Normal': {
                'X': data['X_norm'],
                'Y': data['Y_norm'],
                'Meta': data['meta_norm'] if 'meta_norm' in data.files else None,
                'color_bg': '#e6ffe6'
            },
            'Anomaly': {
                'X': data['X_anom'],
                'Y': data['Y_anom'],
                'Meta': data['meta_anom'] if 'meta_anom' in data.files else None,
                'color_bg': '#ffe6e6'
            }
        }
        
        self.marked_indices = {
            'Normal': set(),
            'Anomaly': set()
        }

        self.model_data = {}
        
        try:
            custom_obj = {'GraphConvolution': GraphConvolution} if 'GraphConvolution' in globals() else {}
            model = load_model(model_path, compile=False, custom_objects=custom_obj)
            
            for key in ['Normal', 'Anomaly']:
                X_data = self.datasets[key]['X']
                Y_data = self.datasets[key]['Y']
                
                if len(X_data) > 0:
                    y_pred = model.predict(X_data, verbose=0)
                    mse_preds = np.mean(np.square(Y_data - y_pred), axis=(1, 2))
                    sorted_indices = np.argsort(mse_preds)[::-1]
                else:
                    y_pred = []
                    mse_preds = []
                    sorted_indices = []

                self.model_data[key] = {
                    'y_pred': y_pred,
                    'mse_preds': mse_preds,
                    'sorted_indices': sorted_indices
                }
            
        except Exception:
            for key in ['Normal', 'Anomaly']:
                n = len(self.datasets[key]['X'])
                self.model_data[key] = {
                    'y_pred': np.zeros_like(self.datasets[key]['Y']),
                    'mse_preds': np.zeros(n),
                    'sorted_indices': np.arange(n)
                }

        self.current_mode = 'Anomaly'
        self.current_step = 0
        
        self.out = widgets.Output()
        
        self.tgl_dataset = widgets.ToggleButtons(
            options=['Normal', 'Anomaly'],
            value='Anomaly',
            description='Датасет:',
            button_style='info'
        )
        self.tgl_dataset.observe(self.on_dataset_change, names='value')

        self.btn_prev = widgets.Button(description="<< Назад", disabled=True)
        self.btn_next = widgets.Button(description="Вперед >>")
        self.btn_delete = widgets.Button(description="Видалити", button_style='warning', icon='trash')
        self.btn_save = widgets.Button(description="Зберегти ВСЕ", button_style='success', icon='save')
        self.lbl_info = widgets.Label(value="")
        self.lbl_file = widgets.HTML(value="")
        
        self.btn_next.on_click(self.on_next)
        self.btn_prev.on_click(self.on_prev)
        self.btn_delete.on_click(self.toggle_delete)
        self.btn_save.on_click(self.save_data)
        
        controls = widgets.HBox([self.btn_prev, self.btn_delete, self.btn_next])
        top_bar = widgets.VBox([self.tgl_dataset, self.lbl_info, self.lbl_file, controls, self.btn_save])
        
        display(top_bar, self.out)
        
        self.render()

    def get_current_data(self):
        ds = self.datasets[self.current_mode]
        md = self.model_data[self.current_mode]
        return ds['X'], ds['Y'], md['y_pred'], md['mse_preds'], md['sorted_indices']

    def get_real_index(self):
        _, _, _, _, sorted_idxs = self.get_current_data()
        if len(sorted_idxs) == 0: return 0
        return sorted_idxs[self.current_step]

    def on_dataset_change(self, change):
        self.current_mode = change.new
        self.current_step = 0
        self.render()

    def render(self):
        X, _, _, mse_preds, _ = self.get_current_data()
        total_samples = len(X)
        
        if total_samples == 0:
            with self.out:
                clear_output()
                print("Датасет порожній")
            return

        real_idx = self.get_real_index()
        current_marked_set = self.marked_indices[self.current_mode]
        is_marked = real_idx in current_marked_set
        
        loss_val = mse_preds[real_idx] if len(mse_preds) > 0 else 0

        meta_array = self.datasets[self.current_mode]['Meta']
        if meta_array is not None and len(meta_array) > real_idx:
            file_info = meta_array[real_idx]
        else:
            file_info = "Метадані відсутні"
        
        self.lbl_info.value = f"Режим: {self.current_mode} | Крок: {self.current_step + 1}/{total_samples} | ID: {real_idx} | MSE: {loss_val:.5f}"
        self.lbl_file.value = f"<b>Джерело:</b> <span style='color:blue;'>{file_info}</span>"
        self.btn_prev.disabled = (self.current_step == 0)
        self.btn_next.disabled = (self.current_step == total_samples - 1)
        
        if is_marked:
            self.btn_delete.description = "Відновити"
            self.btn_delete.button_style = 'danger'
            self.btn_delete.icon = 'undo'
        else:
            self.btn_delete.description = "Видалити"
            self.btn_delete.button_style = 'warning'
            self.btn_delete.icon = 'trash'

        with self.out:
            clear_output(wait=True)
            self.plot_filmstrip(real_idx, is_marked)

    def plot_filmstrip(self, index, is_marked):
        X, Y, y_pred, _, _ = self.get_current_data()
        
        input_seq = X[index, :, :51]
        target_seq = Y[index, :, :51]
        pred_seq = y_pred[index, :, :51] 

        fig, axes = plt.subplots(6, 5, figsize=(15, 18))
        axes_flat = axes.flatten() 
        
        base_bg = self.datasets[self.current_mode]['color_bg']
        bg_color = '#bfbfbf' if is_marked else base_bg
        
        fig.patch.set_facecolor(bg_color)

        coords = [input_seq, target_seq]
        if pred_seq.shape[-1] > 0: coords.append(pred_seq)
            
        min_vals = []
        max_vals = []
        
        for c in coords:
            if c.size > 0:
                min_vals.append(np.min(c[:, 0::3]))
                min_vals.append(np.min(c[:, 1::3]))
                max_vals.append(np.max(c[:, 0::3]))
                max_vals.append(np.max(c[:, 1::3]))
            
        if min_vals:
            x_min, x_max = min(min_vals) - 0.2, max(max_vals) + 0.2
            y_min, y_max = min(min_vals) - 0.2, max(max_vals) + 0.2
        else:
            x_min, x_max, y_min, y_max = -1, 1, -1, 1
        
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10),
            (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]

        for i, ax in enumerate(axes_flat):
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(bg_color)

            if i < 25:
                pose = input_seq[i]
                x, y = pose[0::3], pose[1::3]
                
                ax.set_title(f"In: {i}", fontsize=8)
                ax.scatter(x, y, c='red' if is_marked else 'blue', s=10)
                for s, e in edges:
                    if s < len(x) and e < len(x):
                        ax.plot([x[s], x[e]], [y[s], y[e]], c='black', lw=1)
            
            else:
                target_idx = i - 25
                if target_idx < len(target_seq):
                    ax.set_title(f"Out: {target_idx}", fontsize=9, fontweight='bold')
                    
                    pose_gt = target_seq[target_idx]
                    xg, yg = pose_gt[0::3], pose_gt[1::3]
                    ax.scatter(xg, yg, c='green', s=25, alpha=0.6)
                    for s, e in edges:
                        if s < len(xg) and e < len(xg):
                            ax.plot([xg[s], xg[e]], [yg[s], yg[e]], c='green', lw=2, alpha=0.5)

                    if target_idx < len(pred_seq):
                        pose_pr = pred_seq[target_idx]
                        if pose_pr.size > 0:
                            xp, yp = pose_pr[0::3], pose_pr[1::3]
                            ax.scatter(xp, yp, c='magenta', s=15, marker='x')
                            for s, e in edges:
                                if s < len(xp) and e < len(xp):
                                    ax.plot([xp[s], xp[e]], [yp[s], yp[e]], c='pink', lw=1, linestyle='--')

            if i == 12 and is_marked: 
                ax.text(0.5, 0.5, "ВИДАЛЕНО", color='red', fontsize=24, fontweight='bold', rotation=45, 
                        ha='center', va='center', transform=ax.transAxes, alpha=0.7)

        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        plt.show()

    def on_next(self, b):
        X, _, _, _, _ = self.get_current_data()
        if self.current_step < len(X) - 1:
            self.current_step += 1
            self.render()

    def on_prev(self, b):
        if self.current_step > 0:
            self.current_step -= 1
            self.render()

    def toggle_delete(self, b):
        idx = self.get_real_index()
        current_set = self.marked_indices[self.current_mode]
        
        if idx in current_set:
            current_set.remove(idx)
        else:
            current_set.add(idx)
        self.render()

    def save_data(self, b):
        with self.out:
            indices_anom = np.arange(len(self.datasets['Anomaly']['X']))
            mask_anom = ~np.isin(indices_anom, list(self.marked_indices['Anomaly']))
            X_anom_clean = self.datasets['Anomaly']['X'][mask_anom]
            Y_anom_clean = self.datasets['Anomaly']['Y'][mask_anom]
            
            meta_anom = self.datasets['Anomaly']['Meta']
            meta_anom_clean = meta_anom[mask_anom] if meta_anom is not None else []
            
            indices_norm = np.arange(len(self.datasets['Normal']['X']))
            mask_norm = ~np.isin(indices_norm, list(self.marked_indices['Normal']))
            X_norm_clean = self.datasets['Normal']['X'][mask_norm]
            Y_norm_clean = self.datasets['Normal']['Y'][mask_norm]
            
            meta_norm = self.datasets['Normal']['Meta']
            meta_norm_clean = meta_norm[mask_norm] if meta_norm is not None else []
            
            np.savez(CLEANED_DIR, 
                     X_norm=X_norm_clean, Y_norm=Y_norm_clean, meta_norm=meta_norm_clean,
                     X_anom=X_anom_clean, Y_anom=Y_anom_clean, meta_anom=meta_anom_clean)
            
            clear_output(wait=True)
            print(f"Збережено: {CLEANED_DIR}")
            print(f"Аномалії: {len(X_anom_clean)} (було {len(indices_anom)})")
            print(f"Норма: {len(X_norm_clean)} (було {len(indices_norm)})")