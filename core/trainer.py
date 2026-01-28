import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error
import numpy as np
import copy
import os
import math
from tqdm import tqdm

from core.logger import Logger
from core.config import config


class EMA:
    def __init__(self, model: nn.Module):
        self.model = model
        self.decay = config.ema.decay
        self.warmup_steps = config.ema.warmup_steps
        self.step = 0
        
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def _get_decay(self) -> float:
        if self.step < self.warmup_steps:
            return min(self.decay, (1 + self.step) / (config.ema.warmup_denominator + self.step))
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


class Trainer: 
    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        checkpoint_dir="checkpoints",
        target_scaler=None,
        feature_scaler=None,
        embedding_dimensions=None,
        log_dir = None
    ):
        if config.device is None:
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(config.device)
        self.logger = Logger(name="Trainer", level=logging.INFO, log_dir=log_dir)

        self.logger.section("Trainer Initialization")
        self.logger.info(f"[Device] Using: {self.device}")
        self.logger.info(f"[GPU] Name: {torch.cuda.get_device_name(0)}")
        self.logger.info(f"[GPU] Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.logger.info(f"[GPU] CUDA Version: {torch.version.cuda} \n")

        self._log_parameters(model)

        self.model = model.to(self.device)
        
        self._compile_model()
        
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.target_scaler = target_scaler
        self.feature_scaler = feature_scaler
        self.embedding_dimensions = embedding_dimensions or []
        self.high_target_weight = config.training.high_target_weight
        self.logger.info(f"[High Target Weight] Using high target weight: {self.high_target_weight}\n")

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.criterion = nn.SmoothL1Loss(reduction='none') 
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
        self.scaler    = GradScaler() if config.training.mixed_precision else None
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode=config.scheduler.mode, factor=config.scheduler.factor, patience=config.scheduler.patience)
        self.global_step = 0
        
        self.ema = None
        if config.ema.enabled:
            self.ema = EMA(self.model)
    
    def _compile_model(self):
        if not config.compile.enabled:
            self.logger.info("[Compile] Model compilation disabled")
            return
        
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 0):
            self.logger.warning(f"[Compile] torch.compile() requires PyTorch 2.0+, current version: {torch.__version__}")
            return
        
        if self.device.type != "cuda":
            self.logger.warning("[Compile] torch.compile() works best with CUDA. Skipping on CPU.")
            return
        
        try:
            self.logger.section("[Model Compilation]")
            self.logger.info(f"[Compile] Mode: {config.compile.mode}")
            self.logger.info(f"[Compile] Backend: {config.compile.backend}")
            self.logger.info(f"[Compile] Dynamic shapes: {config.compile.dynamic}")
            self.logger.info(f"[Compile] Full graph: {config.compile.fullgraph}")
            
            self.model = torch.compile(
                self.model,
                mode=config.compile.mode,
                backend=config.compile.backend,
                fullgraph=config.compile.fullgraph,
                dynamic=config.compile.dynamic
            )
            
            self.logger.info("[Compile] Model compiled successfully!")
            self.logger.info("[Compile] Note: First few iterations will be slower due to compilation\n")
            
        except Exception as e:
            self.logger.warning(f"[Compile] Failed to compile model: {e}")
            self.logger.warning("[Compile] Continuing with uncompiled model\n")
          
    def _log_parameters(self, model):
        self.logger.section("[Model Parameter Counts]")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.subsection(f"Full Model - Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
        total_params = sum(p.numel() for p in model.tokenizer.parameters())
        trainable_params = sum(p.numel() for p in model.tokenizer.parameters() if p.requires_grad)
        self.logger.info(f"[Tokenizer] Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
        total_params = sum(p.numel() for p in model.invoice_encoder.parameters())
        trainable_params = sum(p.numel() for p in model.invoice_encoder.parameters() if p.requires_grad)
        self.logger.info(f"[Invoice Encoder] Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
        total_params = sum(p.numel() for p in model.sequence_encoder.parameters())
        trainable_params = sum(p.numel() for p in model.sequence_encoder.parameters() if p.requires_grad)
        self.logger.info(f"[Sequence Encoder] Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
        total_params = sum(p.numel() for p in model.temporal_attention.parameters())
        trainable_params = sum(p.numel() for p in model.temporal_attention.parameters() if p.requires_grad)
        self.logger.info(f"[Temporal Attention] - Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
        total_params = sum(p.numel() for p in model.head_days.parameters())
        trainable_params = sum(p.numel() for p in model.head_days.parameters() if p.requires_grad)
        self.logger.info(f"[Prediction Head] Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
        self.logger.info("")

    def _weighted_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = self.criterion(preds, targets).view(-1)

        if self.high_target_weight and self.target_scaler is not None and self.high_target_weight > 0:
            den_targets = np.expm1(self.target_scaler.inverse_transform(targets.detach().cpu().numpy().reshape(-1, 1))).reshape(-1)
            den_targets = torch.from_numpy(den_targets).to(self.device).float()
            mean_den = den_targets.mean().clamp_min(1e-8)
            weights = 1.0 + self.high_target_weight * (den_targets / mean_den)
            weighted = losses * weights
            return weighted.mean()

        return losses.mean()
          
    def save_checkpoint(self, epoch, metric, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_metric': metric,
            'ema_state_dict': self.ema.state_dict() if self.ema else None,
            'target_scaler': self.target_scaler,
            'feature_scaler': self.feature_scaler,
            'embedding_dimensions': list(self.embedding_dimensions) if self.embedding_dimensions is not None else [],
        }
        
        last_path = os.path.join(self.checkpoint_dir, config.paths.last_checkpoint_name)
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, config.paths.best_model_name)
            torch.save(state, best_path)
            
            if self.ema:
                ema_path = os.path.join(self.checkpoint_dir, config.paths.best_model_ema_name)
                self.ema.apply_shadow()
                torch.save(self.model.state_dict(), ema_path)
                self.ema.restore()
            
            if self.logger:
                self.logger.info(f"[Checkpoint] New best model saved")
    
    def _backward_step(self, loss):
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.training.max_grad_norm)
            self.scaler.step(self.optimizer)   
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.training.max_grad_norm)
            self.optimizer.step()

        self.global_step += 1

        if self.ema:
            self.ema.update()
    
    def train_epoch(self, epoch=None):
        self.model.train()
        running_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        
        if config.overfit.enabled:
            single_batch = next(iter(self.train_loader))
            loop = tqdm(range(len(self.train_loader)), desc=f"Train Epoch {epoch} (Overfit Single Batch)")
            for _ in loop:
                categorical_features, continuous_features, targets, lengths = single_batch
                
                categorical_features = categorical_features.to(self.device, non_blocking=True)
                continuous_features = continuous_features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type=self.device.type, enabled=config.training.mixed_precision):
                    preds = self.model(categorical_features, continuous_features, lengths)
                    target_tensor = targets.view(-1)
                    loss = self._weighted_loss(preds, target_tensor)
                
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
                
                with autocast(device_type=self.device.type, enabled=config.training.mixed_precision):
                    preds = self.model(categorical_features, continuous_features, lengths)
                    target_tensor = targets.view(-1)
                    loss = self._weighted_loss(preds, target_tensor)
                
                self._backward_step(loss)
                running_loss += loss.detach()
                num_batches += 1
        
        average_loss = (running_loss / max(num_batches, 1)).item()
        if self.logger:
            self.logger.log_scalar("Loss/train", average_loss, epoch)
            self.logger.log_scalar("LR", self.optimizer.param_groups[0]['lr'], epoch)
        
        return average_loss
    
    def _metrics(self, den_targets, den_preds, average_loss):
        mae    = np.mean(np.abs(den_preds - den_targets))
        rmse   = np.sqrt(np.mean((den_preds - den_targets) ** 2))
        ss_res = np.sum((den_targets - den_preds) ** 2)
        ss_tot = np.sum((den_targets - np.mean(den_targets)) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
        p50    = np.percentile(np.abs(den_preds - den_targets), 50)
        p90    = np.percentile(np.abs(den_preds - den_targets), 90)
        p99    = np.percentile(np.abs(den_preds - den_targets), 99)

        std_error = np.std(den_preds - den_targets)
        mape      = mean_absolute_percentage_error(den_targets, den_preds)
        medae     = median_absolute_error(den_targets, den_preds)
        max_error = np.max(np.abs(den_targets - den_preds))

        error_bin_0_5      = np.mean(np.abs(den_targets - den_preds) <= 5) * 100
        error_bin_5_10     = np.mean((np.abs(den_targets - den_preds) > 5) & (np.abs(den_targets - den_preds) <= 10)) * 100
        error_bin_10_15    = np.mean((np.abs(den_targets - den_preds) > 10) & (np.abs(den_targets - den_preds) <= 15)) * 100
        error_bin_15_20    = np.mean((np.abs(den_targets - den_preds) > 15) & (np.abs(den_targets - den_preds) <= 20)) * 100
        error_bin_20_25    = np.mean((np.abs(den_targets - den_preds) > 20) & (np.abs(den_targets - den_preds) <= 25)) * 100
        error_bin_above_25 = np.mean(np.abs(den_targets - den_preds) > 25) * 100

        error_target_0_5      = np.mean(np.abs(den_targets - den_preds)[den_targets.flatten() <= 5]) if np.any(den_targets.flatten() <= 5) else np.nan
        error_target_5_10     = np.mean(np.abs(den_targets - den_preds)[(den_targets.flatten() > 5) & (den_targets.flatten() <= 10)]) if np.any((den_targets.flatten() > 5) & (den_targets.flatten() <= 10)) else np.nan
        error_target_10_15    = np.mean(np.abs(den_targets - den_preds)[(den_targets.flatten() > 10) & (den_targets.flatten() <= 15)]) if np.any((den_targets.flatten() > 10) & (den_targets.flatten() <= 15)) else np.nan
        error_target_15_20    = np.mean(np.abs(den_targets - den_preds)[(den_targets.flatten() > 15) & (den_targets.flatten() <= 20)]) if np.any((den_targets.flatten() > 15) & (den_targets.flatten() <= 20)) else np.nan
        error_target_20_25    = np.mean(np.abs(den_targets - den_preds)[(den_targets.flatten() > 20) & (den_targets.flatten() <= 25)]) if np.any((den_targets.flatten() > 20) & (den_targets.flatten() <= 25)) else np.nan
        error_target_above_25 = np.mean(np.abs(den_targets - den_preds)[den_targets.flatten() > 25]) if np.any(den_targets.flatten() > 25) else np.nan

        metrics = {
            'loss': average_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'std_error': std_error,
            'p50': p50,
            'p90': p90,
            'p99': p99,
            'mape': mape,
            'medae': medae,
            'max_error': max_error,
            'error_bin_0_5': error_bin_0_5,
            'error_bin_5_10': error_bin_5_10,
            'error_bin_10_15': error_bin_10_15,
            'error_bin_15_20': error_bin_15_20,
            'error_bin_20_25': error_bin_20_25,
            'error_bin_above_25': error_bin_above_25,
            'error_target_0_5': error_target_0_5,
            'error_target_5_10': error_target_5_10,
            'error_target_10_15': error_target_10_15,
            'error_target_15_20': error_target_15_20,
            'error_target_20_25': error_target_20_25,
            'error_target_above_25': error_target_above_25
        }

        return metrics

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
            loss = self.criterion(preds, target_tensor).mean()
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

        metrics = self._metrics(den_targets, den_preds, average_loss)
        
        return metrics
    
    def fit(self):
        torch.cuda.empty_cache()
        
        self.logger.section("Model Training")
        self.logger.subsection("Training Progress")
        
        best_rmse = float('inf')
        best_model_state = None
        best_ema_state = None
        
        patience_counter = 0
        for epoch in range(1, config.training.epochs + 1):
            train_loss = self.train_epoch(epoch=epoch)
            validation_metrics = self.evaluate(self.validation_loader, use_ema=True)
            validation_rmse = validation_metrics['rmse']
            
            self.scheduler.step(validation_rmse)
            
            self.logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={validation_metrics['loss']:.4f} | MAE={validation_metrics['mae']:.4f} | RMSE={validation_rmse:.4f} | R2={validation_metrics['r2']:.4f} | StdErr={validation_metrics['std_error']:.4f}")
            if validation_rmse < best_rmse:
                best_rmse = validation_rmse
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_ema_state = copy.deepcopy(self.ema.state_dict()) if self.ema else None
                patience_counter = 0
                self.save_checkpoint(epoch, validation_rmse, is_best=True)
                self.logger.info(f" New Best Model: RMSE={validation_rmse:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config.training.patience:
                    self.logger.warning(f"[Early Stopping] Training halted at epoch {epoch} (patience={config.training.patience}). Best RMSE: {best_rmse:.4f}")
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
    
