import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error
import numpy as np
import itertools
import copy
from tqdm.auto import tqdm


class Trainer: 
    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        target_scaler=None,
        logger = None,
        config=None,
    ):
        self.logger = logger
        self.config = config 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader      = train_loader
        self.validation_loader = validation_loader
        self.target_scaler     = target_scaler
    
        self.high_target_weight = self.config.training.high_target_weight
  
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        self.scaler    = GradScaler() if self.config.training.mixed_precision else None

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=self.config.scheduler.scheduler_mode,
            factor=self.config.scheduler.scheduler_factor,
            patience=self.config.scheduler.scheduler_patience,
        )
        
        self.global_step = 0
        self.checkpoint = None

        self.logger.section("Trainer Initialization")
        self.logger.info(f"[High Target Weight] Using high target weight: {self.high_target_weight}\n")
        self.log_parameters(model)

    def log_parameters(self, model):
        self.logger.section("[Model Parameter Counts]")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.subsection(f"Full Model - Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

        children = []
        for name, module in model.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            children.append((name, total, trainable))

        children.sort(key=lambda x: x[1], reverse=True)

        for name, total, trainable in children:
            self.logger.info(f"[{name}] Total parameters: {total:,}, Trainable: {trainable:,}")

        self.logger.info("")

    def weighted_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = self.criterion(preds, targets).view(-1)

        if self.high_target_weight and self.target_scaler is not None and self.high_target_weight > 0:
            den_targets = np.expm1(self.target_scaler.inverse_transform(targets.detach().cpu().numpy().reshape(-1, 1))).reshape(-1)
            den_targets = torch.from_numpy(den_targets).to(self.device).float()
            mean_den = den_targets.mean().clamp_min(1e-8)
            weights = 1.0 + self.high_target_weight * (den_targets / mean_den)
            weighted = losses * weights
            return weighted.mean()

        return losses.mean()
              
    def backward(self, loss):
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            self.scaler.step(self.optimizer)   
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            self.optimizer.step()

        self.global_step += 1

    def train_epoch(self, epoch=None):
        self.model.train()
        running_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
   
        if self.config.overfit.overfit_single_batch:
            single_batch = next(iter(self.train_loader))
            batch_iterable = itertools.repeat(single_batch, len(self.train_loader))
            loop = tqdm(batch_iterable, desc=f"Train Epoch {epoch} (Overfit Single Batch)", total=len(self.train_loader))
        else:
            loop = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")

        for batch in loop:
            categorical_features, continuous_features, targets, lengths = batch

            categorical_features = categorical_features.to(self.device, non_blocking=True)
            continuous_features  = continuous_features.to(self.device, non_blocking=True)
            targets              = targets.to(self.device, non_blocking=True)
            lengths              = lengths.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.config.training.mixed_precision):
                preds         = self.model(categorical_features, continuous_features, lengths)
                target_tensor = targets.view(-1)
                loss          = self.weighted_loss(preds, target_tensor)

            self.backward(loss)
            running_loss += loss.detach()
            num_batches += 1
        
        average_loss = (running_loss / max(num_batches, 1)).item()    
        return average_loss
    
    def compute_metrics(self, den_targets, den_preds, average_loss=None) -> dict:
        den_preds   = np.asarray(den_preds).flatten()
        den_targets = np.asarray(den_targets).flatten()

        mae = float(np.mean(np.abs(den_preds - den_targets)))
        rmse = float(np.sqrt(np.mean((den_preds - den_targets) ** 2)))
        ss_res = float(np.sum((den_targets - den_preds) ** 2))
        ss_tot = float(np.sum((den_targets - np.mean(den_targets)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
        std = float(np.std(den_targets - den_preds))
        medae = float(median_absolute_error(den_targets, den_preds))
        max_error = float(np.max(np.abs(den_targets - den_preds)))
        p50 = np.percentile(np.abs(den_targets - den_preds), 50)
        p90 = np.percentile(np.abs(den_targets - den_preds), 90)
        p95 = np.percentile(np.abs(den_targets - den_preds), 95)

        abs_err = np.abs(den_targets - den_preds)
        error_bin_0_5 = float(np.mean(abs_err <= 5) * 100)
        error_bin_5_10 = float(np.mean((abs_err > 5) & (abs_err <= 10)) * 100)
        error_bin_10_15 = float(np.mean((abs_err > 10) & (abs_err <= 15)) * 100)
        error_bin_15_20 = float(np.mean((abs_err > 15) & (abs_err <= 20)) * 100)
        error_bin_20_25 = float(np.mean((abs_err > 20) & (abs_err <= 25)) * 100)
        error_bin_above_25 = float(np.mean(abs_err > 25) * 100)

        def mean_target_range(low, high=None):
            mask = (den_targets > low) & (den_targets <= high)
            vals = abs_err[mask]
            return float(np.mean(vals)) if len(vals) > 0 else float('nan')

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'std': std,
            'p50': p50,
            'p90': p90,
            'p95': p95,
            'medae': medae,
            'max_error': max_error,
            'loss': average_loss,
            
            '0_5': error_bin_0_5,
            '5_10': error_bin_5_10,
            '10_15': error_bin_10_15,
            '15_20': error_bin_15_20,
            '20_25': error_bin_20_25,
            'above_25': error_bin_above_25,
    
            '0_5': mean_target_range(0, 5),
            '5_10': mean_target_range(5, 10),
            '10_15': mean_target_range(10, 15),
            '15_20': mean_target_range(15, 20),
            '20_25': mean_target_range(20, 25),
            'above_25': mean_target_range(25, 30),
        }
        
        return metrics

    @torch.no_grad()
    def evaluate(self, loader):
        all_preds = []
        all_targets = []
        running_loss = torch.tensor(0.0, device=self.device)
        
        num_batches = 0
        for categorical_features, continuous_features, targets, lengths in loader:
            categorical_features = categorical_features.to(self.device, non_blocking=True)
            continuous_features  = continuous_features.to(self.device, non_blocking=True)
            targets              = targets.to(self.device, non_blocking=True)
            lengths              = lengths.to(self.device, non_blocking=True)
            
            preds         = self.model(categorical_features, continuous_features, lengths)
            target_tensor = targets.view(-1)
            loss          = self.criterion(preds, target_tensor).mean()
            running_loss += loss.detach()
            
            num_batches += 1
            all_preds.append(preds.cpu())
            all_targets.append(target_tensor.cpu())
             
        average_loss = (running_loss / max(num_batches, 1)).item()

        all_preds_tensor = torch.cat(all_preds, dim=0).numpy()
        all_targets_tensor = torch.cat(all_targets, dim=0).numpy()
        
        den_targets = np.expm1(self.target_scaler.inverse_transform(all_targets_tensor.reshape(-1, 1)))
        den_preds   = np.expm1(self.target_scaler.inverse_transform(all_preds_tensor.reshape(-1, 1)))

        den_preds   = np.clip(den_preds, 0, None)
        den_targets = np.clip(den_targets, 0, None)

        metrics = self.compute_metrics(den_targets, den_preds, average_loss)
        
        return metrics
    
    def fit(self):
        torch.cuda.empty_cache()
        
        self.logger.section("Model Training")
        self.logger.subsection("Training Progress")
        
        best_rmse = float('inf')
        best_model_state = None
        
        patience_counter = 0
        for epoch in range(1, self.config.training.epochs + 1):
            train_loss         = self.train_epoch(epoch=epoch)
            validation_metrics = self.evaluate(self.validation_loader)
            validation_rmse    = validation_metrics['rmse']
            
            self.scheduler.step(validation_rmse)
            
            self.logger.info(
                f"Epoch {epoch}:\n"
                f"  Train Loss = {train_loss:.4f}\n"
                f"  Val   Loss = {validation_metrics['loss']:.4f}\n"
                f"  MAE        = {validation_metrics['mae']:.4f} | RMSE = {validation_rmse:.4f}\n"
                f"  R2         = {validation_metrics['r2']:.4f} | StdErr = {validation_metrics['std']:.4f}\n"
                f"  P50 = {validation_metrics['p50']:.4f} | P90 = {validation_metrics['p90']:.4f} | P95 = {validation_metrics['p95']:.4f} \n"
            )
            
            if validation_rmse < best_rmse:
                best_rmse = validation_rmse
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                self.logger.info(f" New Best Model: RMSE={validation_rmse:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.patience:
                    self.logger.warning(f"[Early Stopping] Training halted at epoch {epoch} (patience={self.config.training.patience}). Best RMSE: {best_rmse:.4f}")
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
           
        return self.model
    
