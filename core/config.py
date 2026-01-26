from dataclasses import dataclass, field
from typing import List

@dataclass
class PathsDetails:
    raw_data             : str = "data/raw_data.parquet"
    train_data           : str = "data/training_data.parquet"
    runs_dir             : str = "Runs"
    checkpoints_dir      : str = "checkpoints"
    best_model_name      : str = "best_model.pth"
    last_checkpoint_name : str = "last_checkpoint.pth"
    best_model_ema_name  : str = "best_model_ema.pth"
    optuna_tuning_dir    : str = "optuna_tuning"
    optuna_results_dir   : str = "optuna_results"

@dataclass
class DataParams:
    test_size : float = 0.10
    val_size  : float = 0.10
    
    random_state : int = 42
    
    sample_frac      : float = 1.0
    
    load_sample_frac : float = 0.01
    user_sample_num  : int = 100
    
    rolling_window_sizes: List[int] = field(default_factory=lambda: [3, 5])
    delay_is_known_value: int = 1
    use_log1p_transform: bool = True
    clip_target_min: float = 0.0

@dataclass
class TemporalFeatureParams:
    days_in_week      : int = 7
    months_in_year    : int = 12
    weekend_start_day : int = 5

@dataclass
class AugmentationParams:
    temporal_cutout_ratio : float = 0.15
    feature_dropout_ratio : float = 0.2
    gaussian_noise_std    : float = 0.05
    default_probability: float = 0.1
    time_warp_probability: float = 0.05
    default_std_deviation: float = 0.1

@dataclass
class Columns:
    
    drop_cols: List[str] = field(default_factory=lambda: [
        'emissaoNotaFiscalData', 'emissaoNotaFiscalDia', 'codigoOSOmie', 'numeroNFSeOmie', 
        'criacaoDataAcrescimoDesconto', 'adyenPspReferencePagamento' 
    ])
    
    date_cols: List[str] = field(default_factory=lambda: [
        'vencimentoData', 'pagamentoData', 'pagamentoData_UTC', 'criacaoData', 
        'competenciaInicioData', 'competenciaFimData', 'atualizacao_dt'
    ])
    
    cat_cols: List[str] = field(default_factory=lambda: [
        'recorrencia_pagamento', 'sexo', 'faixa_idade_resumida',
        'veiculo_modelo', 'pacoteNome', 'formaPagamento',
        'lugar', 'regiao', 'produto_categoria',
        'venc_dayofweek', 'venc_quarter', 'venc_is_weekend', 
        'venc_is_month_start', 'venc_is_month_end', 
        'is_first_invoice', 'is_improving', 'is_first_contract', 
        'forma_pagamento_caucao', 
    ])

    cont_cols: List[str] = field(default_factory=lambda: [
        'quantidadeDiarias', 'valor_caucao_brl', 'valor_brl',
        'venc_dayofweek_sin', 'venc_dayofweek_cos',
        'venc_day_sin', 'venc_day_cos',
        'venc_month_sin', 'venc_month_cos',
        'venc_day', 'venc_month',
        'hist_mean_delay', 'hist_std_delay', 'hist_max_delay',
        'last_delay', 'delay_trend',
        'seq_position', 'seq_position_norm', 'days_since_last_invoice',
        'rolling_mean_delay_3', 'rolling_max_delay_3',
        'rolling_mean_delay_5', 'rolling_max_delay_5',
        'parcela_position', 'parcela_position_norm', 'parcela_total',
        'num_contracts', 'contract_mean_delay',
        'value_ratio', 'hist_mean_value',
        'grace_period', 'total_billed', 'total_paid',
        'delay_clipped', 'valor_caucao_entrada_brl', 'pacoteDuracao', 'delay_is_known'
    ])

    target_cols: List[str] = field(default_factory=lambda: [
        'target_days_to_payment'
    ])
    
    user_id_col          : str = 'usuarioId'
    contract_id_col      : str = 'locacaoId'
    order_col            : str = 'ordem_parcela'
    
    criation_date_col    : str = 'criacaoData'
    due_date_col         : str = 'vencimentoData'
    payment_date_col     : str = 'pagamentoData'
    delay_col            : str = 'Dias_atraso'
    
    category_col         : str = 'categoria'
    category_filter      : str = 'aluguel'
    billed_value_col     : str = 'valor_brl'
    payed_value_col      : str = 'valor_pago_brl'
    security_deposit_col : str = 'valor_caucao_brl'
    
    sort_cols: List[str] = field(default_factory=lambda: ['usuarioId', 'criacaoData', 'vencimentoData'])
    group_cols: List[str] = field(default_factory=lambda: ['usuarioId'])
    
    delay_clipped_col: str = 'delay_clipped'
    delay_is_known_col: str = 'delay_is_known'
    target_col_name: str = 'target_days_to_payment'

    no_scale_cols: List[str] = field(default_factory=lambda: ['delay_is_known'])

@dataclass
class ModelParams:
    batch_size: int = 32
    max_seq_len: int = 50
    min_seq_len: int = 5
    
    epochs: int = 30
    lr: float = 3e-4
    dropout: float = 0.05
    weight_decay: float = 1e-4
    patience: int = 5
    
    mixed_precision: bool = True
    max_grad_norm: float = 1.0
   
    num_workers: int = 0
    pin_memory: bool =  False
    persistent_workers: bool = False
    prefetch_factor: int = 0
    
    hidden_dim: int = 128
    n_heads: int = 4
    num_invoice_layers: int = 2
    num_sequence_layers: int = 3

    scheduler_mode: str = 'min'
    scheduler_factor: float = 0.1
    scheduler_patience: int = 10
    
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 2000
    ema_warmup_denominator: int = 10
    
    drop_path_rate: float = 0.05
    
    use_augmentation: bool = False
    augment_prob: float = 0.1
    
    embedding_dropout: float = 0.05
    periodic_sigma: float = 1.0
    
    device: str = None  
   
    rope_base: float = 10000.0
    
    categorical_padding_value: int = 0
    continuous_padding_value: float = 0.0
    
    overfit_epochs: int = 1000
    overfit_patience: int = 1000
    overfit_dropout: float = 0.0
    overfit_weight_decay: float = 0.0
    overfit_num_users: int = 3
    overfit_min_lenghth: int = 2
    overfit_test_size: float = 0.3
    overfit_val_size: float = 0.3
    overfit_use_ema: bool = False
    overfit_use_augmentation: bool = False
    overfit_mixed_precision: bool = False


    overfit_single_batch: bool = False

@dataclass
class OptunaParams:
    study_name: str = "payment_default_study"
    storage: str = None
    n_trials: int = 100
    n_startup_trials: int = 5
    timeout: float = None
    pruning_warmup_epochs: int = 2
    early_stopping_patience: int = 8
    max_epochs: int = 25
    
    learning_rate_min: float = 1e-4
    learning_rate_max: float = 5e-3
    
    hidden_dim_options: List[int] = field(default_factory=lambda: [64, 96, 128, 160, 192])
    
    num_invoice_layers_min: int = 1
    num_invoice_layers_max: int = 3
    
    num_sequence_layers_min: int = 2
    num_sequence_layers_max: int = 5
    
    num_heads_options: List[int] = field(default_factory=lambda: [4, 8])
    
    dropout_min: float = 0.05
    dropout_max: float = 0.3

    weight_decay_min: float = 1e-5
    weight_decay_max: float = 1e-3
    
    periodic_sigma_min: float = 0.5
    periodic_sigma_max: float = 2.0
 
  

@dataclass
class Config:
    paths:        PathsDetails = field(default_factory=PathsDetails)
    data:         DataParams = field(default_factory=DataParams)
    temporal:     TemporalFeatureParams = field(default_factory=TemporalFeatureParams)
    augmentation: AugmentationParams = field(default_factory=AugmentationParams)
    columns:      Columns = field(default_factory=Columns)
    model:        ModelParams = field(default_factory=ModelParams)
    optuna:       OptunaParams = field(default_factory=OptunaParams)

config = Config()
