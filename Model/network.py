import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PeriodicEmbedding(nn.Module):
    def __init__(self, n_features, d_embedding, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.coefficients = nn.Parameter(torch.randn(n_features, d_embedding // 2) * sigma)
        
    def forward(self, x):
        # x: (B, n_features)
        x_proj = x.unsqueeze(-1) * self.coefficients.unsqueeze(0) * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PLREmbedding(nn.Module):
    def __init__(self, n_features, d_embedding, n_bins=8):
        super().__init__()
        self.n_bins = n_bins
        self.linear = nn.Linear(n_features * n_bins, n_features * d_embedding)
        self.d_embedding = d_embedding
        self.n_features = n_features
        
        self.bin_boundaries = nn.Parameter(torch.linspace(-3, 3, n_bins).unsqueeze(0).repeat(n_features, 1))
        
    def forward(self, x):
        B = x.shape[0]
        
        x_expanded = x.unsqueeze(-1)  # (B, n_features, 1)
        boundaries = self.bin_boundaries.unsqueeze(0)  # (1, n_features, n_bins)
        
        plr_encoding = F.relu(1 - torch.abs(x_expanded - boundaries))  # (B, n_features, n_bins)
        plr_flat = plr_encoding.view(B, -1)  # (B, n_features * n_bins)
        
        out = self.linear(plr_flat)  # (B, n_features * d_embedding)
        return out.view(B, self.n_features, self.d_embedding)


class FeatureTokenizer(nn.Module):
    def __init__(self, cardinalities, n_cont, d_token, use_plr=True, n_bins=16):
        super().__init__()
        
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(c, d_token) for c in cardinalities
        ])
        
        self.emb_dropout = nn.Dropout(0.1)
    
        self.n_cont = n_cont
        self.d_token = d_token
        
        if n_cont > 0:
            if use_plr:
                # PLR embedding for continuous features (state-of-the-art for tabular)
                self.cont_embedding = PLREmbedding(n_cont, d_token, n_bins=n_bins)
            else:
                # Fallback to simple linear
                self.cont_weights = nn.Parameter(torch.empty(1, n_cont, d_token))
                self.cont_bias = nn.Parameter(torch.empty(1, n_cont, d_token))
                self._init_cont_params()
            self.use_plr = use_plr
        
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        
    def _init_cont_params(self):
        nn.init.kaiming_uniform_(self.cont_weights, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.cont_weights.size(2))
        nn.init.uniform_(self.cont_bias, -bound, bound)
        
    def forward(self, x_cat, x_cont):
        B = x_cat.shape[0]
        
        cat_tokens = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_tokens.append(emb(x_cat[:, i])) 
        
        if cat_tokens:
            x_cat_t = torch.stack(cat_tokens, dim=1)
            x_cat_t = self.emb_dropout(x_cat_t)
        else:
            x_cat_t = torch.empty(B, 0, self.d_token, device=x_cat.device)
        
        if self.n_cont > 0:
            if self.use_plr:
                x_cont_t = self.cont_embedding(x_cont)
            else:
                x_cont_t = x_cont.unsqueeze(-1) * self.cont_weights + self.cont_bias
        else:
            x_cont_t = torch.empty(B, 0, self.d_token, device=x_cont.device)
            
        cls_t = self.cls_token.expand(B, -1, -1)
        
        x = torch.cat([cls_t, x_cat_t, x_cont_t], dim=1)
        return x


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model, d_hidden=None, dropout=0.1):
        super().__init__()
        d_hidden = d_hidden or d_model * 2
        
        self.layer1 = nn.Linear(d_model, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_model)
        self.gate = nn.Linear(d_hidden, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        h = F.gelu(self.layer1(x))
        h = self.dropout(h)
        
        out = self.layer2(h)
        gate = torch.sigmoid(self.gate(h))
        
        return self.norm(residual + gate * out)


class CrossFeatureAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        attn_out, _ = self.attention(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_out))


class Model(nn.Module):
    def __init__(self, embedding_dims, n_cont, outcome_dim, hidden_dim=64, n_blocks=3, 
                 dropout=0.15, n_heads=4, use_grn=True, use_cross_attention=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_cat = len(embedding_dims)
        self.n_cont = n_cont
        self.use_grn = use_grn
        self.use_cross_attention = use_cross_attention
        
        cardinalities = [c[0] for c in embedding_dims]
        
        self.tokenizer = FeatureTokenizer(cardinalities, n_cont, hidden_dim, use_plr=True, n_bins=8)
        
        if use_grn:
            self.feature_grn = GatedResidualNetwork(hidden_dim, dropout=dropout)
        
        if use_cross_attention and n_cont > 0 and len(cardinalities) > 0:
            self.cross_attn = CrossFeatureAttention(hidden_dim, n_heads=2, dropout=dropout)
     
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=int(hidden_dim * 4), 
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LN for better gradient flow
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks, enable_nested_tensor=False)
        
        self.use_multiscale_pooling = True
        pool_dim = hidden_dim * 3 if self.use_multiscale_pooling else hidden_dim
        
        self.head_default = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.feature_importance = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x_cat, x_cont):
        B = x_cat.shape[0]
        
        # Tokenize features
        x = self.tokenizer(x_cat, x_cont)  # (B, 1 + n_cat + n_cont, hidden_dim)
        
        # Apply GRN for feature selection
        if self.use_grn:
            x = self.feature_grn(x)
        
        # Cross-attention between cat and cont features
        if self.use_cross_attention and self.n_cont > 0 and self.n_cat > 0:
            cat_tokens = x[:, 1:1+self.n_cat, :]  # Skip CLS
            cont_tokens = x[:, 1+self.n_cat:, :]
            
            # Cont attends to cat features
            cont_enhanced = self.cross_attn(cont_tokens, cat_tokens)
            x = torch.cat([x[:, :1+self.n_cat, :], cont_enhanced], dim=1)
        
        # Main transformer processing
        x_processed = self.transformer(x)
        
        # Multi-scale pooling
        if self.use_multiscale_pooling:
            cls_output = x_processed[:, 0, :]  # CLS token
            mean_output = x_processed[:, 1:, :].mean(dim=1)  # Mean of feature tokens
            max_output = x_processed[:, 1:, :].max(dim=1)[0]  # Max of feature tokens
            pooled = torch.cat([cls_output, mean_output, max_output], dim=-1)
        else:
            pooled = x_processed[:, 0, :]
        
        # Prediction
        logits = self.head_default(pooled)
        return logits

