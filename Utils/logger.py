import logging
import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir="logs", name="logger"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        fh = logging.FileHandler(os.path.join(self.log_dir, f'{name}.log'))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))

    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
        
    def error(self, message):
        self.logger.error(message)
    
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def log_figure(self, tag, figure, step):
        self.writer.add_figure(tag, figure, step)
        
    def close(self):
        self.writer.close()
