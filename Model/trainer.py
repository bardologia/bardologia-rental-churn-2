import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
import numpy as np
import time
import copy
import os
import gc
from tqdm import tqdm
from Utils.logger import Logger


class FocalLoss(nn.Module):
    """
    Focal Loss for extreme class imbalance.
    Reduces the relative loss for well-classified examples, focusing on hard negatives.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha  # Weight for positive class
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting for class balance
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply focal and alpha weights
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss - different gamma for positives and negatives.
    Particularly effective for highly imbalanced datasets (~1% positive).
    
    Key idea: Apply hard negative mining (high gamma_neg) while preserving
    all positive samples (low gamma_pos). Also add explicit pos_weight.
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, pos_weight=10.0):
        super().__init__()
        self.gamma_neg = gamma_neg  # Higher for hard negative mining
        self.gamma_pos = gamma_pos  # 0 = don't down-weight any positives
        self.clip = clip  # Probability margin for negative samples
        self.pos_weight = pos_weight  # Explicit weight for positive class
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Asymmetric clipping for negatives (ignore easy negatives)
        probs_neg = (probs + self.clip).clamp(max=1)
        
        # Calculate losses separately
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))
        
        # Apply different gamma - key: gamma_pos=0 keeps all positive samples
        loss_pos = loss_pos * ((1 - probs) ** self.gamma_pos) * self.pos_weight
        loss_neg = loss_neg * (probs_neg ** self.gamma_neg)
        
        loss = -loss_pos - loss_neg
        return loss.mean()

class Trainer:
    def __init__(self, model, data_module, lr=1e-3, epochs=20, patience=5, mixed_precision=True, device=None, weight_decay=1e-5, checkpoint_dir="checkpoints", logger: Logger = None, max_grad_norm=1.0, scheduler_factor=0.3, scheduler_patience=4, min_lr=1e-6, loss_type='focal', focal_alpha=0.75, focal_gamma=2.0):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dm = data_module
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.loss_type = loss_type
        self.optimal_threshold = 0.3  # Start lower for imbalanced data
        
        # Loss function selection for imbalanced data
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            if self.logger:
                self.logger.info(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        elif loss_type == 'asymmetric':
            # Tuned for extreme imbalance (~1% positive)
            self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, pos_weight=15.0)
            if self.logger:
                self.logger.info("Using Asymmetric Loss (gamma_neg=4, gamma_pos=0, pos_weight=15)")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            if self.logger:
                self.logger.info("Using standard BCE Loss")
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.patience = patience
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm
        self.scaler = GradScaler() if mixed_precision and self.device.type == 'cuda' else None
        
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_lr = min_lr
        self.scheduler = None

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            if self.logger:
                self.logger.info(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint.get('epoch', 0), checkpoint.get('best_metric', 0)
        else:
            if self.logger:
                self.logger.info(f"No checkpoint found at {path}")
            return 0, 0

    def save_checkpoint(self, epoch, metric, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': metric,
        }
      
        last_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
            if self.logger:
                self.logger.info(f"New best model saved to {best_path} with AUC: {metric:.4f}")

    def log_model_internals(self, epoch):
        if not self.logger:
            return
            
        for name, param in self.model.named_parameters():
            self.logger.log_histogram(f"Weights/{name}", param, epoch)
            if param.grad is not None:
                self.logger.log_histogram(f"Gradients/{name}", param.grad, epoch)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        n_pos, n_total = 0, 0
        
        loop = tqdm(dataloader, desc=f"Train Epoch {epoch}")
        for batch_idx, (x_cat, x_cont, y) in enumerate(loop):
            x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            with autocast(device_type=self.device.type, enabled=self.mixed_precision and self.device.type == 'cuda'):
                logits = self.model(x_cat, x_cont)
                
                # Single target output
                target = y.view(-1, 1) if y.dim() == 1 else y[:, 0].unsqueeze(1)
                
                # Track class balance in batch
                n_pos += target.sum().item()
                n_total += target.numel()
                
                # Use configured loss (Focal/Asymmetric/BCE)
                loss = self.criterion(logits, target)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=current_lr)
            
            # Clean up batch tensors
            del x_cat, x_cont, y, logits, target, loss
        
        avg_loss = total_loss / len(dataloader)
        pos_rate = n_pos / n_total if n_total > 0 else 0
        
        if self.logger:
            self.logger.log_scalar("Train/Loss", avg_loss, epoch)
            self.logger.log_scalar("Train/PositiveRate", pos_rate, epoch)
            self.log_model_internals(epoch)
            
        return avg_loss

    def find_optimal_threshold(self, targets, predictions):
        precision, recall, thresholds = precision_recall_curve(targets, predictions)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        return optimal_threshold, f1_scores[best_idx]

    def validate(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        loop = tqdm(dataloader, desc=f"Val Epoch {epoch}")
        with torch.no_grad():
            for x_cat, x_cont, y in loop:
                x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device)
                
                logits = self.model(x_cat, x_cont)
                target = y.view(-1, 1) if y.dim() == 1 else y[:, 0].unsqueeze(1)
                
                loss = self.criterion(logits, target)

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
                all_preds.extend(torch.sigmoid(logits).view(-1).cpu().numpy())
                all_targets.extend(target.view(-1).cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        

        try:
            roc_auc = roc_auc_score(all_targets, all_preds)
            pr_auc = average_precision_score(all_targets, all_preds)
            self.optimal_threshold, best_f1 = self.find_optimal_threshold(all_targets, all_preds)
            preds_binary = (all_preds >= self.optimal_threshold).astype(int)
            f1 = f1_score(all_targets, preds_binary, zero_division=0)
            
            tp = ((preds_binary == 1) & (all_targets == 1)).sum()
            fp = ((preds_binary == 1) & (all_targets == 0)).sum()
            fn = ((preds_binary == 0) & (all_targets == 1)).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
        except Exception as e:
            roc_auc, pr_auc, f1, precision, recall = 0.5, 0.0, 0.0, 0.0, 0.0
            self.optimal_threshold = 0.5
        
        aucs = {'default': pr_auc}  
        
        if self.logger:
            self.logger.log_scalar("Val/Loss", avg_loss, epoch)
            self.logger.log_scalar("Val/ROC_AUC", roc_auc, epoch)
            self.logger.log_scalar("Val/PR_AUC", pr_auc, epoch)
            self.logger.log_scalar("Val/F1", f1, epoch)
            self.logger.log_scalar("Val/Precision", precision, epoch)
            self.logger.log_scalar("Val/Recall", recall, epoch)
            self.logger.log_scalar("Val/OptimalThreshold", self.optimal_threshold, epoch)
            self.logger.log_histogram("Val/Predictions", all_preds, epoch)
            
            val_pos_rate = all_targets.mean()
            self.logger.log_scalar("Val/PositiveRate", val_pos_rate, epoch)
        
        return avg_loss, aucs

    def fit(self):
        msg = f"Training on device: {self.device}"
   
        self.logger.info(msg)
     
        if self.mixed_precision and self.device.type == 'cuda':
            self.logger.info("Mixed Precision Enabled (AMP)")
        
        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=self.scheduler_factor, 
            patience=self.scheduler_patience, 
            min_lr=self.min_lr
        )
        
        best_auc = 0
        patience_counter = 0
                    
        for epoch in range(self.epochs):
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            train_loss = self.train_epoch(train_loader, epoch)
            
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            val_loss, aucs = self.validate(val_loader, epoch)

            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
            avg_auc = np.mean(list(aucs.values()))
            old_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step(avg_auc)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Log LR after scheduler step
            if self.logger:
                self.logger.log_scalar("Train/LearningRate", new_lr, epoch)
                if new_lr != old_lr:
                    self.logger.info(f"LR reduced: {old_lr:.2e} -> {new_lr:.2e}")
            
            if self.device.type == 'cuda' and self.logger:
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                self.logger.log_scalar("System/GPU_Allocated_GB", allocated, epoch)
                self.logger.log_scalar("System/GPU_Reserved_GB", reserved, epoch)

            avg_auc = np.mean(list(aucs.values()))
            
            if self.logger:
                self.logger.log_scalar("Val/Mean_AUC", avg_auc, epoch)
            
            is_best = avg_auc > best_auc
            if is_best:
                best_auc = avg_auc
                flag = "*"
                patience_counter = 0
                self.save_checkpoint(epoch, best_auc, is_best=True)
            else:
                flag = ""
                patience_counter += 1
                self.save_checkpoint(epoch, best_auc, is_best=False)
                
            log_msg = (f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"AUC: {avg_auc:.4f} | LR: {new_lr:.2e} {flag}")
            
            if self.logger:
                self.logger.info(log_msg)
            else:
                print(log_msg)
                
            if patience_counter >= self.patience:
                if self.logger:
                    self.logger.info(f"Early stop triggered after {epoch+1} epochs.")
                else:
                    print(f"Early stop triggered after {epoch+1} epochs.")
                break
            
        final_msg = "Traning Complete.\n" + f"Best Mean AUC: {best_auc:.4f}"
        if self.logger:
            self.logger.info(final_msg)
        else:
            print(final_msg)
        
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_path):
             self.load_checkpoint(best_path)
             
        return self.model

    def test(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        if self.logger:
            self.logger.info(f"Starting Testing Phase (using threshold: {self.optimal_threshold:.4f})...")

        loop = tqdm(dataloader, desc="Testing")
        with torch.no_grad():
            for x_cat, x_cont, y in loop:
                x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device)
                
                logits = self.model(x_cat, x_cont)
                target = y.view(-1, 1) if y.dim() == 1 else y[:, 0].unsqueeze(1)
                
                loss = self.criterion(logits, target)

                total_loss += loss.item()
                
                all_preds.extend(torch.sigmoid(logits).view(-1).cpu().numpy())
                all_targets.extend(target.view(-1).cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(all_targets, all_preds)
            pr_auc = average_precision_score(all_targets, all_preds)
            
            # Use optimal threshold from validation
            preds_binary = (all_preds >= self.optimal_threshold).astype(int)
            f1 = f1_score(all_targets, preds_binary, zero_division=0)
            
            # Detailed metrics at optimal threshold
            tp = ((preds_binary == 1) & (all_targets == 1)).sum()
            fp = ((preds_binary == 1) & (all_targets == 0)).sum()
            fn = ((preds_binary == 0) & (all_targets == 1)).sum()
            tn = ((preds_binary == 0) & (all_targets == 0)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            
            # Also find test-optimal threshold for comparison
            test_opt_threshold, test_best_f1 = self.find_optimal_threshold(all_targets, all_preds)
            
        except Exception as e:
            roc_auc, pr_auc, f1, precision, recall, specificity = 0.5, 0.0, 0.0, 0.0, 0.0, 0.0
            test_opt_threshold, test_best_f1 = 0.5, 0.0
        
        aucs = {'default': pr_auc}
        
        # Positive rate in test set
        test_pos_rate = all_targets.mean()
        
        if self.logger:
            self.logger.log_scalar("Test/Loss", avg_loss, 0)
            self.logger.log_scalar("Test/ROC_AUC", roc_auc, 0)
            self.logger.log_scalar("Test/PR_AUC", pr_auc, 0)
            self.logger.log_scalar("Test/F1", f1, 0)
            self.logger.log_scalar("Test/Precision", precision, 0)
            self.logger.log_scalar("Test/Recall", recall, 0)
            self.logger.log_histogram("Test/Predictions", all_preds, 0)
             
            results_msg = (
                f"\n{'='*50}\n"
                f"TEST RESULTS (Extreme Imbalance: {test_pos_rate*100:.2f}% positive)\n"
                f"{'='*50}\n"
                f"  Loss:       {avg_loss:.4f}\n"
                f"  ROC AUC:    {roc_auc:.4f}\n"
                f"  PR AUC:     {pr_auc:.4f} (key metric for imbalanced data)\n"
                f"{'='*50}\n"
                f"At Val-Optimal Threshold ({self.optimal_threshold:.4f}):\n"
                f"  F1 Score:   {f1:.4f}\n"
                f"  Precision:  {precision:.4f}\n"
                f"  Recall:     {recall:.4f}\n"
                f"  Specificity:{specificity:.4f}\n"
                f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n"
                f"{'='*50}\n"
                f"Test-Optimal Threshold: {test_opt_threshold:.4f} (F1={test_best_f1:.4f})\n"
                f"{'='*50}"
            )
            self.logger.info(results_msg)
        
        return aucs
