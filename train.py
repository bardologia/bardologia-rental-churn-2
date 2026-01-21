import sys
import os
import torch
import numpy as np
import cProfile
import pstats
import io
from datetime import datetime

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from model.core import DataPipeline
from model.data import DataModule
from model.network import Model
from model.trainer import Trainer
from configs.config import config
from utils.logger import Logger


def main(enable_profiler: bool = False):
    profiler = None
    if enable_profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    
    project_root = current_dir
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "Runs", run_id)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    model_save_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    raw_path = os.path.join(project_root, config.paths.raw_data)
    data_path = os.path.join(project_root, config.paths.train_data)
    
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        pipeline = DataPipeline()
        pipeline.run(raw_path, data_path)
        print(f"Generated new training data: {data_path}")
    
    logger = Logger(log_dir=run_dir, name="PaymentDefaultPrediction")
    logger.section("Experiment Initialization")
    logger.info(f"[Experiment] Run ID: {run_id}")
    logger.info(f"[Experiment] Run directory: {run_dir}")
    if enable_profiler:
        logger.info(f"[Profiler] cProfile enabled")
    
    try:
        logger.subsection("Data Module Setup")
        data_module = DataModule(
            data_path,
            batch_size=config.model.batch_size,
            num_workers=config.model.num_workers,
            pin_memory=config.model.pin_memory,
            logger=logger,
            max_sequence_length=config.model.max_seq_len,
            min_sequence_length=config.data.min_sequence_length
        )
        data_module.prepare_data()
        
        logger.subsection("Model Architecture")
        model = Model(
            embedding_dimensions=data_module.embedding_dimensions,
            num_continuous=len(data_module.continuous_columns),
            hidden_dimension=config.model.hidden_dim,
            num_invoice_layers=config.model.num_invoice_layers,
            num_sequence_layers=config.model.n_blocks,
            num_heads=config.model.n_heads,
            dropout=config.model.dropout,
            drop_path_rate=config.model.drop_path_rate,
            use_temporal_attention=config.model.use_temporal_attention,
            use_temperature_scaling=config.model.use_temperature_scaling,
            max_sequence_length=config.model.max_seq_len
        )
        
        num_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        num_layers = config.model.num_invoice_layers + config.model.n_blocks
        
        logger.log_model_summary(
            model_name="SOTAModel (Hierarchical Transformer)",
            num_params=num_parameters,
            architecture_info={
                "Embedding Dimensions": len(data_module.embedding_dimensions),
                "Continuous Features": len(data_module.continuous_columns),
                "Hidden Dimension": config.model.hidden_dim,
                "Invoice Layers": config.model.num_invoice_layers,
                "Sequence Layers": config.model.n_blocks,
                "Attention Heads": config.model.n_heads,
                "Dropout": config.model.dropout,
                "DropPath Rate": config.model.drop_path_rate,
                "Max Sequence Length": config.model.max_seq_len,
                "Temporal Attention": "Enabled" if config.model.use_temporal_attention else "Disabled",
                "Temperature Scaling": "Enabled" if config.model.use_temperature_scaling else "Disabled",
            }
        )
        
        logger.subsection("Trainer Configuration")
        trainer = Trainer(
            model, data_module,
            epochs=config.model.epochs,
            learning_rate=config.model.lr,
            weight_decay=config.model.weight_decay,
            logger=logger,
            patience=config.model.patience,
            mixed_precision=config.model.mixed_precision,
            checkpoint_dir=checkpoint_dir,
            max_grad_norm=config.model.max_grad_norm,
            scheduler_factor=config.model.scheduler_factor,
            scheduler_patience=config.model.scheduler_patience,
            min_learning_rate=config.model.min_lr,
            loss_type=config.model.loss_type,
            use_compile=False  
        )
        
        best_model = trainer.fit()
        
        test_loader = data_module.test_dataloader()
        test_metrics = trainer.test(test_loader)
        
        logger.section("Model Persistence")
        torch.save(best_model.state_dict(), model_save_path)
        logger.info(f"[Save] Best model saved to: {model_save_path}")
        
        device = next(model.parameters()).device
        model.eval()
        all_probabilities = []
        all_targets = []
        with torch.inference_mode():  # Faster than no_grad()
            for categorical_features, continuous_features, targets, lengths in test_loader:
                categorical_features = categorical_features.to(device, non_blocking=True)
                continuous_features = continuous_features.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                logits = model(categorical_features, continuous_features, lengths)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                all_probabilities.append(probabilities)
                all_targets.append(targets.numpy())
        
        probs_path = os.path.join(run_dir, 'test_probs_targets.npz')
        np.savez(
            probs_path,
            probs=np.concatenate(all_probabilities, axis=0),
            targets=np.concatenate(all_targets, axis=0)
        )
        logger.info(f"[Save] Test probabilities saved to: {probs_path}")
        
        logger.section("Final Results Summary")
        logger.metrics_table(
            headers=["Target", "AUC-ROC", "AUC-PR", "F1-Score"],
            rows=[
                [name, 
                 f"{test_metrics.get(f'{name}_auc_roc', 0):.4f}",
                 f"{test_metrics.get(f'{name}_auc_pr', 0):.4f}",
                 f"{test_metrics.get(f'{name}_f1', 0):.4f}"]
                for name in config.columns.target_cols
            ],
            title="Test Set Performance"
        )
        
        if profiler is not None:
            profiler.disable()
            
            logger.section("Profiler Results")
            
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats(30)
            
            logger.info("[Profiler] Top 30 functions by cumulative time:")
            for line in stream.getvalue().split('\n'):
                if line.strip():
                    logger.info(line)
            
            profile_path = os.path.join(run_dir, "profile_results.prof")
            profiler.dump_stats(profile_path)
            logger.info(f"[Profiler] Detailed profile saved to: {profile_path}")
            logger.info("[Profiler] Visualize with: snakeviz profile_results.prof")
        
    except Exception as e:
        logger.error(f"[Error] Training failed: {e}")
        import traceback
        traceback.print_exc()
        if profiler is not None:
            profiler.disable()
    finally:
        logger.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Run with cProfile profiler")
    args = parser.parse_args()
    
    main(enable_profiler=args.profile)
