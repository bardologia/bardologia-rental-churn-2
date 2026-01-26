import logging
import os
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, log_dir="logs", name="experiment", level="INFO", enable_tensorboard=True):
        self.log_dir = log_dir
        self.name = name
        self.start_time = datetime.now()
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
        
        self.enable_tensorboard = enable_tensorboard
        self.writer = None
        if enable_tensorboard and log_dir:
            tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        self._log_experiment_header()
    
    def _log_experiment_header(self):
        self.logger.info(f"[Experiment] {self.name}")
        self.logger.info(f"[Start] {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.log_dir:
            self.logger.info(f"[Log Dir] {os.path.abspath(self.log_dir)}")
        if self.enable_tensorboard:
            self.logger.info(f"[TensorBoard] Enabled")

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
    
    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_figure(self, tag: str, figure, step: int):
        if self.writer:
            self.writer.add_figure(tag, figure, step)
    
    def log_text(self, tag: str, text: str, step: int):
        if self.writer:
            self.writer.add_text(tag, text, step)
    
    def log_hyperparams(self, hparam_dict: dict, metric_dict: dict = None):
        if self.writer:
            self.writer.add_hparams(hparam_dict, metric_dict or {})
    
    def log_model_summary(self, model_name: str = None, num_params: int = None, 
                          architecture_info: dict = None, model=None):
        self.logger.info(f"  > Model Architecture")
        
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.logger.info(f"    Total Parameters:     {total_params:,}")
            self.logger.info(f"    Trainable Parameters: {trainable_params:,}")
            self.logger.info(f"    Model Size (MB):      {total_params * 4 / (1024**2):.2f}")
            
            for name, module in model.named_children():
                params = sum(p.numel() for p in module.parameters())
                self.logger.info(f"      {name:<28} : {params:,}")
        else:
            if model_name:
                self.logger.info(f"    Model: {model_name}")
            if num_params is not None:
                self.logger.info(f"    Trainable Parameters: {num_params:,}")
                self.logger.info(f"    Model Size (MB): {num_params * 4 / (1024**2):.2f}")
            
            if architecture_info:
                max_key_len = max(len(str(k)) for k in architecture_info.keys())
                for key, value in architecture_info.items():
                    self.logger.info(f"    {str(key):<{max_key_len}} : {value}")
    
    def log_training_start(self, num_epochs: int, batch_size: int, learning_rate: float, 
                           train_samples: int, val_samples: int, test_samples: int):
        self.logger.info("")
        self.logger.info(">>> TRAINING")
        self.logger.info(f"    Epochs: {num_epochs} | Batch: {batch_size} | LR: {learning_rate:.2e}")
        self.logger.info(f"    Train: {train_samples:,} | Val: {val_samples:,} | Test: {test_samples:,}")
    
    def log_epoch_results(self, epoch: int, total_epochs: int, train_loss: float, 
                          val_loss: float, metrics: dict = None, val_metrics: dict = None,
                          learning_rate: float = None,
                          is_best: bool = False, elapsed_time: float = None):
        
        status = "★" if is_best else ""
        time_str = f"({elapsed_time:.1f}s)" if elapsed_time else ""
        
        metrics_to_use = metrics if metrics is not None else val_metrics
        lr = learning_rate if learning_rate is not None else (metrics_to_use.get('LR', 0) if metrics_to_use else 0)
        
        metrics_str = ""
        if metrics_to_use:
            metrics_parts = []
            for key, value in metrics_to_use.items():
                if key == 'LR':
                    continue
                if isinstance(value, float):
                    metrics_parts.append(f"{key}: {value:.4f}")
            metrics_str = " | ".join(metrics_parts)
        
        self.logger.info(
            f"Epoch [{epoch:03d}/{total_epochs:03d}] {time_str} "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | {metrics_str} | LR: {lr:.2e} {status}"
        )
    
    def log_evaluation_results(self, dataset_name: str = None, phase: str = None, metrics: dict = None):
        title = dataset_name if dataset_name else (str(phase).upper() if phase else "Evaluation")
        self.logger.info(f"  > {title}")
        
        if metrics:
            max_key_len = max(len(str(k)) for k in metrics.keys())
            for key, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"    {str(key):<{max_key_len}} : {value:.4f}")
                else:
                    self.logger.info(f"    {str(key):<{max_key_len}} : {value}")
    
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
        
        if self.writer:
            self.writer.close()
        
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
