import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import copy
import os
from tqdm import tqdm

from core.logger import Logger
from core.config import config


class EMA:
    def __init__(self, model: nn.Module):
        self.model = model
        self.decay = config.model.ema_decay
        self.warmup_steps = config.model.ema_warmup_steps
        self.step = 0
        
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def _get_decay(self) -> float:
        if self.step < self.warmup_steps:
            return min(self.decay, (1 + self.step) / (10 + self.step))
        return self.decay
    
    @torch.no_grad()
    def update(self):
        decay = self._get_decay()
        self.step += 1
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        return {
            'shadow': self.shadow,
            'step': self.step,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.step = state_dict['step']
        self.decay = state_dict.get('decay', self.decay)


class AsymmetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_negative = config.loss.asymmetric_gamma_negative
        self.gamma_positive = config.loss.asymmetric_gamma_positive
        self.clip = config.loss.asymmetric_clip
        
    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        probabilities_clipped = probabilities.clamp(min=self.clip)

        positive_loss = targets * torch.log(probabilities_clipped.clamp(min=1e-8))
        if self.gamma_positive > 0:
            positive_loss = positive_loss * ((1 - probabilities) ** self.gamma_positive)
        
        negative_probabilities = (probabilities - self.clip).clamp(min=0)
        negative_loss = (1 - targets) * torch.log((1 - negative_probabilities).clamp(min=1e-8))
        if self.gamma_negative > 0:
            negative_loss = negative_loss * (negative_probabilities ** self.gamma_negative)
        
        loss = -(positive_loss + negative_loss)
        return loss.mean()
        

class Trainer: 
    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        checkpoint_dir="checkpoints",
        target_scaler=None,
        feature_scaler=None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger(name="Trainer", level=logging.INFO)

        self.logger.section("Trainer Initialization")
        self.logger.info(f"[Device] Using: {self.device}")
        self.logger.info(f"[GPU] Name: {torch.cuda.get_device_name(0)}")
        self.logger.info(f"[GPU] Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.logger.info(f"[GPU] CUDA Version: {torch.version.cuda} \n")

        self.model = model.to(self.device)
        
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.target_scaler = target_scaler
        self.feature_scaler = feature_scaler

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.criterion = nn.SmoothL1Loss() 
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.model.lr, weight_decay=config.model.weight_decay)
        self.scaler    = GradScaler() if config.model.mixed_precision else None
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.model.scheduler_t0,
            T_mult=config.model.scheduler_t_mult,
            eta_min=config.model.min_lr
        )
        
        self.ema = None
        if config.model.use_ema:
            self.ema = EMA(self.model)
          
    def save_checkpoint(self, epoch, metric, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': metric,
            'ema_state_dict': self.ema.state_dict() if self.ema else None,
            'target_scaler': self.target_scaler,
            'feature_scaler': self.feature_scaler,
        }
        
        last_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
            
            if self.ema:
                ema_path = os.path.join(self.checkpoint_dir, 'best_model_ema.pth')
                self.ema.apply_shadow()
                torch.save(self.model.state_dict(), ema_path)
                self.ema.restore()
            
            if self.logger:
                self.logger.info(f"[Checkpoint] New best model saved")
    
    def _backward_step(self, loss):
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.model.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.model.max_grad_norm)
            self.optimizer.step()
        
        if self.ema:
            self.ema.update()
    
    def train_epoch(self, epoch=None):
        self.model.train()
        running_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        
        if config.model.overfit_single_batch:
            single_batch = next(iter(self.train_loader))
            loop = tqdm(range(len(self.train_loader)), desc=f"Train Epoch {epoch} (Overfit Single Batch)")
            for _ in loop:
                categorical_features, continuous_features, targets, lengths = single_batch
                categorical_features = categorical_features.to(self.device, non_blocking=True)
                continuous_features = continuous_features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type=self.device.type, enabled=config.model.mixed_precision):
                    preds = self.model(categorical_features, continuous_features, lengths)
                    target_tensor = targets.view(-1)
                    loss = self.criterion(preds, target_tensor)
                
                self._backward_step(loss)
                
                running_loss += loss.detach()
                num_batches += 1
        else:
            loop = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
            
            for batch_index, (categorical_features, continuous_features, targets, lengths) in enumerate(loop):
                categorical_features = categorical_features.to(self.device, non_blocking=True)
                continuous_features = continuous_features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type=self.device.type, enabled=config.model.mixed_precision):
                    preds = self.model(categorical_features, continuous_features, lengths)
                    target_tensor = targets.view(-1)
                    loss = self.criterion(preds, target_tensor)
                
                self._backward_step(loss)
                
                running_loss += loss.detach()
                num_batches += 1
        
        average_loss = (running_loss / max(num_batches, 1)).item()
        if self.logger:
            self.logger.log_scalar("Loss/train", average_loss, epoch)
            self.logger.log_scalar("LR", self.optimizer.param_groups[0]['lr'], epoch)
        
        return average_loss
    
    @torch.no_grad()
    def evaluate(self, loader, use_ema=True):
        self.model.eval()
        if use_ema and self.ema:
            self.ema.apply_shadow()
        all_preds = []
        all_targets = []
        running_loss = torch.tensor(0.0, device=self.device)
        
        num_batches = 0
        for categorical_features, continuous_features, targets, lengths in loader:
            categorical_features = categorical_features.to(self.device, non_blocking=True)
            continuous_features = continuous_features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)
            
            preds = self.model(categorical_features, continuous_features, lengths)
            target_tensor = targets.view(-1)
            loss = self.criterion(preds, target_tensor)
            running_loss += loss.detach()
            
            num_batches += 1
            all_preds.append(preds.cpu())
            all_targets.append(target_tensor.cpu())
        
        if use_ema and self.ema:
            self.ema.restore()
        
        average_loss = (running_loss / max(num_batches, 1)).item()

        all_preds_tensor = torch.cat(all_preds, dim=0).numpy()
        all_targets_tensor = torch.cat(all_targets, dim=0).numpy()
        
        den_targets = np.expm1(self.target_scaler.inverse_transform(all_targets_tensor.reshape(-1, 1)))
        den_preds   = np.expm1(self.target_scaler.inverse_transform(all_preds_tensor.reshape(-1, 1)))

        den_preds   = np.clip(den_preds, 0, None)
        den_targets = np.clip(den_targets, 0, None)

        mae = np.mean(np.abs(den_preds - den_targets))
        rmse = np.sqrt(np.mean((den_preds - den_targets) ** 2))
        ss_res = np.sum((den_targets - den_preds) ** 2)
        ss_tot = np.sum((den_targets - np.mean(den_targets)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

        std_error = np.std(den_preds - den_targets)
        metrics = {
            'loss': average_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'std_error': std_error
        }
        
        return metrics
    
    def fit(self):
        torch.cuda.empty_cache()
        
        self.logger.section("Model Training")
        self.logger.subsection("Training Progress")
        
        best_rmse = float('inf')
        best_model_state = None
        best_ema_state = None
        
        patience_counter = 0
        for epoch in range(1, config.model.epochs + 1):
            train_loss = self.train_epoch(epoch=epoch)
            validation_metrics = self.evaluate(self.validation_loader, use_ema=True)
            validation_rmse = validation_metrics['rmse']
            self.scheduler.step()
            
            self.logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={validation_metrics['loss']:.4f} | MAE={validation_metrics['mae']:.4f} | RMSE={validation_rmse:.4f}")
            if validation_rmse < best_rmse:
                best_rmse = validation_rmse
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_ema_state = copy.deepcopy(self.ema.state_dict()) if self.ema else None
                patience_counter = 0
                self.save_checkpoint(epoch, validation_rmse, is_best=True)
                self.logger.info(f" New Best Model: RMSE={validation_rmse:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config.model.patience:
                    self.logger.warning(f"[Early Stopping] Training halted at epoch {epoch} (patience={config.model.patience}). Best RMSE: {best_rmse:.4f}")
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if best_ema_state is not None:
                self.ema.load_state_dict(best_ema_state)
        
        self.logger.log_experiment_summary(
            best_metrics={
                "Best RMSE": best_rmse,
            },
            notes=f"Training completed com sucesso. Checkpoints salvos em {self.checkpoint_dir}"
        )
        
        return self.model
    
