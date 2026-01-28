from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathsConfig:
    raw_data             : str = "data/raw_data.parquet"
    train_data           : str = "data/training_data.parquet"
    runs_dir             : str = "runs"
    checkpoints_dir      : str = "checkpoints"
    best_model_name      : str = "best_model.pth"
    last_checkpoint_name : str = "last_checkpoint.pth"
    best_model_ema_name  : str = "best_model_ema.pth"
    optuna_tuning_dir    : str = "optuna_tuning"
    optuna_results_dir   : str = "optuna_results"


@dataclass
class DataSplitConfig:
    test_size    : float = 0.10
    val_size     : float = 0.10
    random_state : int = 42


@dataclass
class DataSamplingConfig:
    sample_frac      : float = 1.0
    load_sample_frac : float = 0.01
    user_sample_num  : Optional[int] = 3000


@dataclass
class TargetConfig:
    target_threshold     : Optional[float] = 30
    delay_is_known_value : int = 1
    use_log1p_transform  : bool = True
    clip_target_min      : float = 0.0


@dataclass
class TemporalFeatureConfig:
    days_in_week      : int = 7
    months_in_year    : int = 12
    weekend_start_day : int = 5


@dataclass
class SequenceFeatureConfig:
    rolling_window_sizes : List[int] = field(default_factory=lambda: [3, 5])


@dataclass
class AugmentationConfig:
    enabled               : bool = False
    probability           : float = 0.1
    temporal_cutout_ratio : float = 0.15
    feature_dropout_ratio : float = 0.2
    gaussian_noise_std    : float = 0.05
    time_warp_probability : float = 0.05


@dataclass
class ModelArchitectureConfig:
    hidden_dim          : int = 128
    n_heads             : int = 4
    num_invoice_layers  : int = 1
    num_sequence_layers : int = 1
    dropout             : float = 0.05
    embedding_dropout   : float = 0.05
    drop_path_rate      : float = 0.05
    periodic_sigma      : float = 1.0
    rope_base           : float = 10000.0


@dataclass
class SequenceConfig:
    max_seq_len                : int = 50
    min_seq_len                : int = 10
    categorical_padding_value  : int = 0
    continuous_padding_value   : float = 0.0


@dataclass
class TrainingConfig:
    batch_size         : int = 512
    epochs             : int = 20
    lr                 : float = 1e-3
    weight_decay       : float = 1e-4
    patience           : int = 6
    max_grad_norm      : float = 3.0
    high_target_weight : float = 0.5
    mixed_precision    : bool = True


@dataclass
class SchedulerConfig:
    mode     : str = 'min'
    factor   : float = 0.5
    patience : int = 3


@dataclass
class EMAConfig:
    enabled            : bool = False
    decay              : float = 0.9999
    warmup_steps       : int = 2000
    warmup_denominator : int = 10


@dataclass
class DataLoaderConfig:
    num_workers : int = 8
    pin_memory  : bool = True


@dataclass
class CompileConfig:
    enabled   : bool = False
    mode      : str = "reduce-overhead"
    backend   : str = "inductor"
    fullgraph : bool = False
    dynamic   : bool = True


@dataclass
class OverfitConfig:
    enabled          : bool = True
    epochs           : int = 100
    patience         : int = 100
    dropout          : float = 0.0
    weight_decay     : float = 0.0
    num_users        : int = 3
    min_seq_len      : int = 2
    test_size        : float = 0.3
    val_size         : float = 0.3
    use_ema          : bool = False
    use_augmentation : bool = False
    mixed_precision  : bool = False


@dataclass
class ColumnConfig:
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
    
    user_id_col     : str = 'usuarioId'
    contract_id_col : str = 'locacaoId'
    order_col       : str = 'ordem_parcela'
    
    criation_date_col : str = 'criacaoData'
    due_date_col      : str = 'vencimentoData'
    payment_date_col  : str = 'pagamentoData'
    delay_col         : str = 'Dias_atraso'
    
    category_col         : str = 'categoria'
    category_filter      : str = 'aluguel'
    billed_value_col     : str = 'valor_brl'
    payed_value_col      : str = 'valor_pago_brl'
    security_deposit_col : str = 'valor_caucao_brl'
    
    sort_cols  : List[str] = field(default_factory=lambda: ['usuarioId', 'criacaoData', 'vencimentoData'])
    group_cols : List[str] = field(default_factory=lambda: ['usuarioId'])
    
    delay_clipped_col  : str = 'delay_clipped'
    delay_is_known_col : str = 'delay_is_known'
    target_col_name    : str = 'target_days_to_payment'
    
    no_scale_cols: List[str] = field(default_factory=lambda: ['delay_is_known'])


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    data_split    : DataSplitConfig = field(default_factory=DataSplitConfig)
    data_sampling : DataSamplingConfig = field(default_factory=DataSamplingConfig)
    target        : TargetConfig = field(default_factory=TargetConfig)
    
    temporal_features : TemporalFeatureConfig = field(default_factory=TemporalFeatureConfig)
    sequence_features : SequenceFeatureConfig = field(default_factory=SequenceFeatureConfig)
    
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    architecture : ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    sequence     : SequenceConfig = field(default_factory=SequenceConfig)
    
    training   : TrainingConfig = field(default_factory=TrainingConfig)
    scheduler  : SchedulerConfig = field(default_factory=SchedulerConfig)
    ema        : EMAConfig = field(default_factory=EMAConfig)
    dataloader : DataLoaderConfig = field(default_factory=DataLoaderConfig)
    compile    : CompileConfig = field(default_factory=CompileConfig)
    overfit    : OverfitConfig = field(default_factory=OverfitConfig)
    
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    
    device: Optional[str] = None


config = Config()
