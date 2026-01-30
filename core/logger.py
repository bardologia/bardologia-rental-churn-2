import logging
import os
import sys
from datetime import datetime
# TensorBoard support removed — logging only
import json
from dataclasses import asdict, is_dataclass


class Logger:
    
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, log_dir="logs", name="experiment", level="INFO", config=None):
        self.log_dir = log_dir
        self.name = name
        self.start_time = datetime.now()
        self.config = config
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        log_level = self.LOG_LEVELS.get(str(level).upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        log_filename = f'{name}_{self.start_time.strftime("%Y%m%d_%H%M%S")}.log'
        if log_dir:
            file_handler = logging.FileHandler(os.path.join(self.log_dir, log_filename), encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)
    
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        self.logger.addHandler(console_handler)
        self._log_experiment_header()
    
    def save_config_file(self, cfg):
        try:
            cfg_dict = asdict(cfg)
            config_filename = f"{self.name}_config_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            config_path = os.path.join(self.log_dir, config_filename)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(cfg_dict, f, indent=2, default=str, ensure_ascii=False)

            self.logger.info(f"  > Configuration saved: {os.path.abspath(config_path)}")
        except Exception as e:
            self.logger.error(f"Failed to serialize config: {e}")

    def _log_experiment_header(self):
        self.logger.info(f"[Experiment] {self.name}")
        self.logger.info(f"[Start] {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.log_dir:
            self.logger.info(f"[Log Dir] {os.path.abspath(self.log_dir)}")

    def section(self, title: str):
        self.logger.info("")
        self.logger.info(f">>> {str(title).upper()}")
    
    def subsection(self, title: str):
        self.logger.info(f"  > {title}")
    
    def experiment_config(self, config_dict: dict, title: str = "Configuration"):
        self.logger.info(f"  > {title}")
        max_key_len = max(len(str(k)) for k in config_dict.keys()) if config_dict else 0
        for key, value in config_dict.items():
            self.logger.info(f"    {str(key):<{max_key_len}} : {value}")
    
    def metrics_table(self, headers: list = None, rows: list = None, title: str = "Metrics", metrics: dict = None, precision: int = 4):
        self.logger.info(f"  > {title}")
        
        if metrics is not None:
            max_key_len = max(len(str(k)) for k in metrics.keys()) if metrics else 0
            for key, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"    {str(key):<{max_key_len}} : {value:.{precision}f}")
                else:
                    self.logger.info(f"    {str(key):<{max_key_len}} : {value}")
        elif headers is not None and rows is not None:
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
            
            header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
            self.logger.info(f"    {header_line}")
            
            for row in rows:
                row_line = " | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row))
                self.logger.info(f"    {row_line}")
    
    def progress(self, current: int, total: int, prefix: str = "", suffix: str = ""):
        percentage = 100 * (current / float(total))
        self.logger.info(f"{prefix} [{current}/{total}] ({percentage:.1f}%) {suffix}")

    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
        
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def log_experiment_summary(self, best_metrics: dict = None, total_epochs: int = None, 
                               early_stopped: bool = False, stopped_epoch: int = None,
                               notes: str = None):
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info("")
        self.logger.info(">>> EXPERIMENT COMPLETE")
        self.logger.info(f"    Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        if total_epochs is not None:
            epochs_run = stopped_epoch if early_stopped else total_epochs
            self.logger.info(f"    Epochs: {epochs_run}{' (early stopped)' if early_stopped else ''}")
        
        if best_metrics:
            for key, value in best_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"    {key}: {value:.4f}")
                else:
                    self.logger.info(f"    {key}: {value}")
        
        if notes:
            self.logger.info(f"    {notes}")
        
    def close(self):
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info(f"[End] Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
