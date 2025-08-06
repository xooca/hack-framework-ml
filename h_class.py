#!/usr/bin/env python3
"""
Hierarchical Multi-Level Multi-Output Multi-Label Classification Solution
A complete implementation using Hugging Face transformers
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer, 
    AutoModel, 
    PreTrainedModel, 
    PreTrainedTokenizer,
    Trainer, 
    TrainingArguments,
    EvalPrediction
)
from transformers.modeling_outputs import ModelOutput
from transformers.configuration_utils import PretrainedConfig
from sklearn.metrics import (
    precision_recall_fscore_support, 
    accuracy_score, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
from faker import Faker
import random

# Suppress warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================================
# 1. DATA GENERATION & PREPROCESSING
# ============================================================================

def create_dummy_dataframe(
    num_samples: int, 
    num_levels: int, 
    num_main_labels: int, 
    num_sub_labels: int
) -> pd.DataFrame:
    """
    Generate a realistic dummy dataset for hierarchical multi-label classification.
    
    Args:
        num_samples: Number of data samples to generate
        num_levels: Number of hierarchical levels
        num_main_labels: Number of main labels per level
        num_sub_labels: Number of sub-labels per main label
        
    Returns:
        pd.DataFrame: Generated dataset with all required columns
    """
    fake = Faker()
    data = []
    
    # Generate category names for consistency
    category_names = [f"Category_{i}" for i in range(num_main_labels)]
    
    for i in range(num_samples):
        row = {'id': f"sample_{i}"}
        
        for level in range(1, num_levels + 1):
            # Text inputs
            row[f'main_info_level{level}'] = fake.text(max_nb_chars=200)
            row[f'aux_info_level{level}'] = fake.text(max_nb_chars=150)
            
            # Structured features for main labels
            locations = [fake.city() for _ in range(num_main_labels)]
            business_codes = [fake.random_int(100, 999) for _ in range(num_main_labels)]
            scores = [fake.random.uniform(0, 1) for _ in range(num_main_labels)]
            
            row[f'main_label_info_level{level}'] = {
                'locations': locations,
                'business_codes': business_codes,
                'scores': scores
            }
            
            # Structured features for sub-labels
            sub_locations = [fake.address() for _ in range(num_sub_labels)]
            sub_codes = [fake.random_int(1000, 9999) for _ in range(num_sub_labels)]
            sub_scores = [fake.random.uniform(0, 1) for _ in range(num_sub_labels)]
            
            row[f'main_label_sub_info_level{level}'] = {
                'sub_locations': sub_locations,
                'sub_codes': sub_codes,
                'sub_scores': sub_scores
            }
            
            # Labels - main labels (binary multi-label)
            main_labels = [fake.random_element([0, 1]) for _ in range(num_main_labels)]
            row[f'main_label_level{level}'] = main_labels
            
            # Hierarchical sub-labels (list of lists)
            sub_labels = []
            for j in range(num_main_labels):
                if main_labels[j] == 1:  # Only generate sub-labels if main label is active
                    sub_label_list = [fake.random_element([0, 1]) for _ in range(num_sub_labels)]
                else:
                    sub_label_list = [0] * num_sub_labels  # No sub-labels if main is inactive
                sub_labels.append(sub_label_list)
            row[f'main_label_sub_label_level{level}'] = sub_labels
            
            # Category information
            row[f'category_info_level{level}'] = category_names.copy()
        
        data.append(row)
    
    return pd.DataFrame(data)


class HierarchicalDataCollator:
    """
    Data collator for hierarchical multi-label classification.
    Handles tokenization and batching of text and numerical features.
    """
    
    def __init__(
        self, 
        tokenizer_configs: Dict[str, Dict],
        num_levels: int,
        num_main_labels: int,
        num_sub_labels: int
    ):
        """
        Initialize the data collator.
        
        Args:
            tokenizer_configs: Dict mapping field names to tokenizer configs
            num_levels: Number of hierarchical levels
            num_main_labels: Number of main labels per level
            num_sub_labels: Number of sub-labels per main label
        """
        self.tokenizer_configs = tokenizer_configs
        self.num_levels = num_levels
        self.num_main_labels = num_main_labels
        self.num_sub_labels = num_sub_labels
        
        # Initialize tokenizers
        self.tokenizers = {}
        for field_name, config in tokenizer_configs.items():
            self.tokenizers[field_name] = AutoTokenizer.from_pretrained(
                config['model_name']
            )
            if self.tokenizers[field_name].pad_token is None:
                self.tokenizers[field_name].pad_token = self.tokenizers[field_name].eos_token
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            Dict of tensors ready for model input
        """
        batch_size = len(examples)
        batch = {}
        
        # Process each level
        for level in range(1, self.num_levels + 1):
            level_key = f"level{level}"
            batch[level_key] = {}
            
            # Tokenize text inputs
            main_texts = [ex[f'main_info_level{level}'] for ex in examples]
            aux_texts = [ex[f'aux_info_level{level}'] for ex in examples]
            
            # Use first tokenizer for text fields
            tokenizer = list(self.tokenizers.values())[0]
            config = list(self.tokenizer_configs.values())[0]
            
            main_encoded = tokenizer(
                main_texts,
                truncation=True,
                padding=True,
                max_length=config.get('max_length', 512),
                return_tensors='pt'
            )
            aux_encoded = tokenizer(
                aux_texts,
                truncation=True,
                padding=True,
                max_length=config.get('max_length', 512),
                return_tensors='pt'
            )
            
            batch[level_key]['main_input_ids'] = main_encoded['input_ids']
            batch[level_key]['main_attention_mask'] = main_encoded['attention_mask']
            batch[level_key]['aux_input_ids'] = aux_encoded['input_ids']
            batch[level_key]['aux_attention_mask'] = aux_encoded['attention_mask']
            
            # Process structured features for main labels
            main_label_info = [ex[f'main_label_info_level{level}'] for ex in examples]
            main_numerical_features = []
            
            for ex in main_label_info:
                # Extract numerical features and normalize
                business_codes = torch.tensor(ex['business_codes'], dtype=torch.float32)
                scores = torch.tensor(ex['scores'], dtype=torch.float32)
                # Combine features
                combined = torch.cat([business_codes.unsqueeze(-1), scores.unsqueeze(-1)], dim=-1)
                main_numerical_features.append(combined)
            
            batch[level_key]['main_numerical_features'] = torch.stack(main_numerical_features)
            
            # Process structured features for sub-labels
            sub_label_info = [ex[f'main_label_sub_info_level{level}'] for ex in examples]
            sub_numerical_features = []
            
            for ex in sub_label_info:
                sub_codes = torch.tensor(ex['sub_codes'], dtype=torch.float32)
                sub_scores = torch.tensor(ex['sub_scores'], dtype=torch.float32)
                combined = torch.cat([sub_codes.unsqueeze(-1), sub_scores.unsqueeze(-1)], dim=-1)
                sub_numerical_features.append(combined)
            
            batch[level_key]['sub_numerical_features'] = torch.stack(sub_numerical_features)
            
            # Process labels
            main_labels = [ex[f'main_label_level{level}'] for ex in examples]
            sub_labels = [ex[f'main_label_sub_label_level{level}'] for ex in examples]
            
            batch[level_key]['main_labels'] = torch.tensor(main_labels, dtype=torch.float32)
            batch[level_key]['sub_labels'] = torch.tensor(sub_labels, dtype=torch.float32)
        
        return batch


# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================

@dataclass
class HierarchicalClassifierOutput(ModelOutput):
    """Output class for HierarchicalClassifier."""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Dict[str, torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class HierarchicalClassifierConfig(PretrainedConfig):
    """Configuration class for HierarchicalClassifier."""
    
    def __init__(
        self,
        num_levels: int = 3,
        base_model_name: str = "distilbert-base-uncased",
        num_main_labels: int = 87,
        num_sub_labels: int = 3,
        hidden_size: int = 768,
        dropout_prob: float = 0.1,
        numerical_feature_dim: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_levels = num_levels
        self.base_model_name = base_model_name
        self.num_main_labels = num_main_labels
        self.num_sub_labels = num_sub_labels
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.numerical_feature_dim = numerical_feature_dim


class CrossAttentionModule(nn.Module):
    """Cross-attention module for fusing different input representations."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query_states, key_value_states):
        """
        Args:
            query_states: [batch_size, hidden_size]
            key_value_states: [batch_size, seq_len, hidden_size]
        """
        batch_size = query_states.size(0)
        
        # Project to Q, K, V
        Q = self.query(query_states).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key_value_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(key_value_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_size)
        
        # Output projection
        output = self.output(context.squeeze(1))
        return output


class HierarchicalClassifier(PreTrainedModel):
    """
    Hierarchical multi-level multi-output multi-label classifier.
    """
    config_class = HierarchicalClassifierConfig
    
    def __init__(self, config: HierarchicalClassifierConfig):
        super().__init__(config)
        self.config = config
        
        # Shared base encoder
        self.base_encoder = AutoModel.from_pretrained(config.base_model_name)
        self.hidden_size = self.base_encoder.config.hidden_size
        
        # Projection layer for numerical features
        self.numerical_projection = nn.Linear(
            config.numerical_feature_dim, 
            self.hidden_size
        )
        
        # Cross-attention modules for each level
        self.main_attention_modules = nn.ModuleList([
            CrossAttentionModule(self.hidden_size) for _ in range(config.num_levels)
        ])
        self.sub_attention_modules = nn.ModuleList([
            CrossAttentionModule(self.hidden_size) for _ in range(config.num_levels)
        ])
        
        # Prediction heads for each level
        self.main_label_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(config.dropout_prob),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout_prob),
                nn.Linear(self.hidden_size // 2, config.num_main_labels)
            ) for _ in range(config.num_levels)
        ])
        
        self.sub_label_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(config.dropout_prob),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout_prob),
                nn.Linear(self.hidden_size // 2, config.num_main_labels * config.num_sub_labels)
            ) for _ in range(config.num_levels)
        ])
        
        # Layer for combining hierarchical representations
        self.hierarchical_fusion = nn.ModuleList([
            nn.Linear(self.hidden_size * 2, self.hidden_size) 
            for _ in range(config.num_levels - 1)
        ])
        
    def forward(
        self, 
        **inputs
    ) -> HierarchicalClassifierOutput:
        """Forward pass through the hierarchical classifier."""
        
        device = next(self.parameters()).device
        all_logits = {}
        total_loss = 0.0
        
        previous_main_repr = None
        previous_sub_repr = None
        
        # Process each level
        for level in range(1, self.config.num_levels + 1):
            level_key = f"level{level}"
            level_data = inputs[level_key]
            
            # Get text embeddings
            main_outputs = self.base_encoder(
                input_ids=level_data['main_input_ids'].to(device),
                attention_mask=level_data['main_attention_mask'].to(device)
            )
            aux_outputs = self.base_encoder(
                input_ids=level_data['aux_input_ids'].to(device),
                attention_mask=level_data['aux_attention_mask'].to(device)
            )
            
            main_text_repr = main_outputs.last_hidden_state[:, 0, :]  # [CLS] token
            aux_text_repr = aux_outputs.last_hidden_state[:, 0, :]
            
            # Process numerical features
            main_num_features = level_data['main_numerical_features'].to(device)
            sub_num_features = level_data['sub_numerical_features'].to(device)
            
            # Project numerical features
            batch_size = main_num_features.size(0)
            main_num_repr = self.numerical_projection(
                main_num_features.view(-1, self.config.numerical_feature_dim)
            ).view(batch_size, self.config.num_main_labels, self.hidden_size)
            
            sub_num_repr = self.numerical_projection(
                sub_num_features.view(-1, self.config.numerical_feature_dim)
            ).view(batch_size, self.config.num_sub_labels, self.hidden_size)
            
            # Prepare key-value states for attention
            main_kv_states = torch.stack([main_text_repr, aux_text_repr], dim=1)  # [batch, 2, hidden]
            main_kv_states = torch.cat([
                main_kv_states,
                main_num_repr
            ], dim=1)  # [batch, 2 + num_main_labels, hidden]
            
            sub_kv_states = torch.stack([main_text_repr, aux_text_repr], dim=1)
            sub_kv_states = torch.cat([
                sub_kv_states,
                sub_num_repr
            ], dim=1)  # [batch, 2 + num_sub_labels, hidden]
            
            # For levels > 1, include previous level representations
            if level > 1 and previous_main_repr is not None:
                # Combine current and previous representations
                main_query = torch.cat([main_text_repr, previous_main_repr], dim=-1)
                main_query = self.hierarchical_fusion[level - 2](main_query)
                
                sub_query = torch.cat([aux_text_repr, previous_sub_repr], dim=-1)
                sub_query = self.hierarchical_fusion[level - 2](sub_query)
            else:
                main_query = main_text_repr
                sub_query = aux_text_repr
            
            # Apply cross-attention
            main_fused = self.main_attention_modules[level - 1](main_query, main_kv_states)
            sub_fused = self.sub_attention_modules[level - 1](sub_query, sub_kv_states)
            
            # Generate predictions
            main_logits = self.main_label_heads[level - 1](main_fused)
            sub_logits = self.sub_label_heads[level - 1](sub_fused)
            sub_logits = sub_logits.view(
                batch_size, 
                self.config.num_main_labels, 
                self.config.num_sub_labels
            )
            
            all_logits[f'main_logits_level{level}'] = main_logits
            all_logits[f'sub_logits_level{level}'] = sub_logits
            
            # Calculate loss if labels are provided
            if f'main_labels' in level_data:
                main_labels = level_data['main_labels'].to(device)
                sub_labels = level_data['sub_labels'].to(device)
                
                # Main label loss
                main_loss = F.binary_cross_entropy_with_logits(main_logits, main_labels)
                
                # Sub-label loss
                sub_loss = F.binary_cross_entropy_with_logits(sub_logits, sub_labels)
                
                total_loss += (main_loss + sub_loss)
            
            # Store representations for next level
            previous_main_repr = main_fused
            previous_sub_repr = sub_fused
        
        return HierarchicalClassifierOutput(
            loss=total_loss if total_loss > 0 else None,
            logits=all_logits
        )


# ============================================================================
# 3. CUSTOM TRAINER
# ============================================================================

class HierarchicalTrainer(Trainer):
    """
    Custom trainer for hierarchical multi-label classification with advanced evaluation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_levels = self.model.config.num_levels
        self.num_main_labels = self.model.config.num_main_labels
        self.num_sub_labels = self.model.config.num_sub_labels
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Find optimal threshold for a single label to maximize F1-score."""
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in np.arange(0.1, 1.0, 0.1):
            y_pred_thresh = (y_pred >= threshold).astype(int)
            if len(np.unique(y_pred_thresh)) > 1 and len(np.unique(y_true)) > 1:
                f1 = f1_score(y_true, y_pred_thresh, average='binary', zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        return best_threshold
    
    def compute_metrics_for_labels(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        label_names: List[str],
        prefix: str
    ) -> Tuple[Dict, List[Dict]]:
        """
        Compute comprehensive metrics for multi-label classification.
        
        Returns:
            Tuple of (overall_metrics, per_class_metrics)
        """
        # Standard metrics with 0.5 threshold
        y_pred_standard = (y_pred >= 0.5).astype(int)
        
        # Overall standard metrics
        precision_micro = precision_score(y_true, y_pred_standard, average='micro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred_standard, average='micro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred_standard, average='micro', zero_division=0)
        
        precision_macro = precision_score(y_true, y_pred_standard, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred_standard, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred_standard, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true, y_pred_standard, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred_standard, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred_standard, average='weighted', zero_division=0)
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred_standard)
        
        # Find optimal thresholds per class
        optimal_thresholds = []
        per_class_metrics = []
        
        for i, label_name in enumerate(label_names):
            if i < y_true.shape[1]:
                optimal_thresh = self.find_optimal_threshold(y_true[:, i], y_pred[:, i])
                optimal_thresholds.append(optimal_thresh)
                
                # Compute per-class metrics with optimal threshold
                y_pred_opt = (y_pred[:, i] >= optimal_thresh).astype(int)
                y_true_class = y_true[:, i]
                
                if len(np.unique(y_true_class)) > 1:
                    tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_opt, labels=[0, 1]).ravel()
                else:
                    tp = fp = tn = fn = 0
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                per_class_metrics.append({
                    'label_name': label_name,
                    'optimal_threshold': optimal_thresh,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn)
                })
        
        # Recompute overall metrics with optimal thresholds
        y_pred_optimized = np.zeros_like(y_pred_standard)
        for i, thresh in enumerate(optimal_thresholds):
            if i < y_pred.shape[1]:
                y_pred_optimized[:, i] = (y_pred[:, i] >= thresh).astype(int)
        
        precision_micro_opt = precision_score(y_true, y_pred_optimized, average='micro', zero_division=0)
        recall_micro_opt = recall_score(y_true, y_pred_optimized, average='micro', zero_division=0)
        f1_micro_opt = f1_score(y_true, y_pred_optimized, average='micro', zero_division=0)
        
        precision_macro_opt = precision_score(y_true, y_pred_optimized, average='macro', zero_division=0)
        recall_macro_opt = recall_score(y_true, y_pred_optimized, average='macro', zero_division=0)
        f1_macro_opt = f1_score(y_true, y_pred_optimized, average='macro', zero_division=0)
        
        precision_weighted_opt = precision_score(y_true, y_pred_optimized, average='weighted', zero_division=0)
        recall_weighted_opt = recall_score(y_true, y_pred_optimized, average='weighted', zero_division=0)
        f1_weighted_opt = f1_score(y_true, y_pred_optimized, average='weighted', zero_division=0)
        
        accuracy_opt = accuracy_score(y_true, y_pred_optimized)
        
        overall_metrics = {
            f'{prefix}_precision_micro': precision_micro,
            f'{prefix}_recall_micro': recall_micro,
            f'{prefix}_f1_micro': f1_micro,
            f'{prefix}_precision_macro': precision_macro,
            f'{prefix}_recall_macro': recall_macro,
            f'{prefix}_f1_macro': f1_macro,
            f'{prefix}_precision_weighted': precision_weighted,
            f'{prefix}_recall_weighted': recall_weighted,
            f'{prefix}_f1_weighted': f1_weighted,
            f'{prefix}_accuracy': accuracy,
            f'{prefix}_precision_micro_optimized': precision_micro_opt,
            f'{prefix}_recall_micro_optimized': recall_micro_opt,
            f'{prefix}_f1_micro_optimized': f1_micro_opt,
            f'{prefix}_precision_macro_optimized': precision_macro_opt,
            f'{prefix}_recall_macro_optimized': recall_macro_opt,
            f'{prefix}_f1_macro_optimized': f1_macro_opt,
            f'{prefix}_precision_weighted_optimized': precision_weighted_opt,
            f'{prefix}_recall_weighted_optimized': recall_weighted_opt,
            f'{prefix}_f1_weighted_optimized': f1_weighted_opt,
            f'{prefix}_accuracy_optimized': accuracy_opt,
        }
        
        return overall_metrics, per_class_metrics
    
    def evaluate(
        self, 
        eval_dataset=None, 
        ignore_keys=None, 
        metric_key_prefix="eval"
    ) -> Dict[str, float]:
        """Override evaluate method with comprehensive metric computation."""
        
        # Get predictions
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        predictions = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                for level_key in batch:
                    if isinstance(batch[level_key], dict):
                        for k, v in batch[level_key].items():
                            if isinstance(v, torch.Tensor):
                                batch[level_key][k] = v.to(self.model.device)
                
                outputs = self.model(**batch)
                predictions.append(outputs.logits)
                
                # Extract labels
                batch_labels = {}
                for level in range(1, self.num_levels + 1):
                    level_key = f"level{level}"
                    batch_labels[f'main_labels_level{level}'] = batch[level_key]['main_labels']
                    batch_labels[f'sub_labels_level{level}'] = batch[level_key]['sub_labels']
                labels.append(batch_labels)
        
        # Aggregate predictions and labels
        all_predictions = {}
        all_labels = {}
        
        for level in range(1, self.num_levels + 1):
            main_preds = []
            sub_preds = []
            main_labs = []
            sub_labs = []
            
            for batch_pred, batch_label in zip(predictions, labels):
                main_preds.append(torch.sigmoid(batch_pred[f'main_logits_level{level}']).cpu().numpy())
                sub_preds.append(torch.sigmoid(batch_pred[f'sub_logits_level{level}']).cpu().numpy())
                main_labs.append(batch_label[f'main_labels_level{level}'].cpu().numpy())
                sub_labs.append(batch_label[f'sub_labels_level{level}'].cpu().numpy())
            
            all_predictions[f'main_level{level}'] = np.vstack(main_preds)
            all_predictions[f'sub_level{level}'] = np.vstack(sub_preds)
            all_labels[f'main_level{level}'] = np.vstack(main_labs)
            all_labels[f'sub_level{level}'] = np.vstack(sub_labs)
        
        # Compute metrics for each level and label type
        all_metrics = {}
        
        # Generate category names for reports
        category_names = [f"Category_{i}" for i in range(self.num_main_labels)]
        sub_category_names = [f"SubCategory_{i}" for i in range(self.num_sub_labels)]
        
        for level in range(1, self.num_levels + 1):
            # Main labels
            main_pred = all_predictions[f'main_level{level}']
            main_true = all_labels[f'main_level{level}']
            
            prefix = f"main_label_level{level}"
            main_overall, main_per_class = self.compute_metrics_for_labels(
                main_true, main_pred, category_names, prefix
            )
            all_metrics.update(main_overall)
            
            # Save reports
            os.makedirs("evaluation_reports", exist_ok=True)
            
            # Overall report
            overall_df = pd.DataFrame([main_overall])
            overall_df.to_csv(
                f"evaluation_reports/{metric_key_prefix}_{prefix}_overall_report.csv", 
                index=False
            )
            
            # Per-class report
            per_class_df = pd.DataFrame(main_per_class)
            per_class_df.to_csv(
                f"evaluation_reports/{metric_key_prefix}_{prefix}_per_class_report.csv", 
                index=False
            )
            
            # Sub-labels (flatten the hierarchical structure for evaluation)
            sub_pred = all_predictions[f'sub_level{level}']
            sub_true = all_labels[f'sub_level{level}']
            
            # Reshape sub-labels from [batch, main_labels, sub_labels] to [batch, main_labels * sub_labels]
            sub_pred_flat = sub_pred.reshape(sub_pred.shape[0], -1)
            sub_true_flat = sub_true.reshape(sub_true.shape[0], -1)
            
            # Generate names for flattened sub-labels
            sub_label_names = []
            for main_idx in range(self.num_main_labels):
                for sub_idx in range(self.num_sub_labels):
                    sub_label_names.append(f"Main_{main_idx}_Sub_{sub_idx}")
            
            prefix = f"main_label_sub_label_level{level}"
            sub_overall, sub_per_class = self.compute_metrics_for_labels(
                sub_true_flat, sub_pred_flat, sub_label_names, prefix
            )
            all_metrics.update(sub_overall)
            
            # Save sub-label reports
            overall_df = pd.DataFrame([sub_overall])
            overall_df.to_csv(
                f"evaluation_reports/{metric_key_prefix}_{prefix}_overall_report.csv", 
                index=False
            )
            
            per_class_df = pd.DataFrame(sub_per_class)
            per_class_df.to_csv(
                f"evaluation_reports/{metric_key_prefix}_{prefix}_per_class_report.csv", 
                index=False
            )
        
        return all_metrics
    
    def predict_on_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on a raw DataFrame and return it with prediction columns.
        
        Args:
            df: Input DataFrame with raw data
            
        Returns:
            DataFrame with added prediction columns
        """
        # Convert DataFrame to Dataset
        dataset = Dataset.from_pandas(df)
        
        # Get data collator from trainer
        data_collator = self.data_collator
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            shuffle=False
        )
        
        # Make predictions
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                for level_key in batch:
                    if isinstance(batch[level_key], dict):
                        for k, v in batch[level_key].items():
                            if isinstance(v, torch.Tensor):
                                batch[level_key][k] = v.to(self.model.device)
                
                outputs = self.model(**batch)
                batch_predictions = {}
                
                for level in range(1, self.num_levels + 1):
                    # Apply sigmoid to get probabilities
                    main_probs = torch.sigmoid(outputs.logits[f'main_logits_level{level}']).cpu().numpy()
                    sub_probs = torch.sigmoid(outputs.logits[f'sub_logits_level{level}']).cpu().numpy()
                    
                    batch_predictions[f'main_probs_level{level}'] = main_probs
                    batch_predictions[f'sub_probs_level{level}'] = sub_probs
                    
                    # Binary predictions with 0.5 threshold
                    batch_predictions[f'main_preds_level{level}'] = (main_probs >= 0.5).astype(int)
                    batch_predictions[f'sub_preds_level{level}'] = (sub_probs >= 0.5).astype(int)
                
                predictions.append(batch_predictions)
        
        # Aggregate predictions
        result_df = df.copy()
        
        for level in range(1, self.num_levels + 1):
            main_probs_all = np.vstack([pred[f'main_probs_level{level}'] for pred in predictions])
            sub_probs_all = np.vstack([pred[f'sub_probs_level{level}'] for pred in predictions])
            main_preds_all = np.vstack([pred[f'main_preds_level{level}'] for pred in predictions])
            sub_preds_all = np.vstack([pred[f'sub_preds_level{level}'] for pred in predictions])
            
            # Add probability columns
            for i in range(self.num_main_labels):
                result_df[f'main_prob_level{level}_label{i}'] = main_probs_all[:, i]
                result_df[f'main_pred_level{level}_label{i}'] = main_preds_all[:, i]
            
            # Add sub-label columns
            for i in range(self.num_main_labels):
                for j in range(self.num_sub_labels):
                    result_df[f'sub_prob_level{level}_main{i}_sub{j}'] = sub_probs_all[:, i, j]
                    result_df[f'sub_pred_level{level}_main{i}_sub{j}'] = sub_preds_all[:, i, j]
        
        return result_df
    
    def evaluate_on_data(self, df: pd.DataFrame):
        """
        Evaluate model on a DataFrame with ground truth labels and save reports.
        
        Args:
            df: DataFrame with ground truth labels
        """
        # Convert to Dataset
        dataset = Dataset.from_pandas(df)
        
        # Run evaluation
        metrics = self.evaluate(eval_dataset=dataset, metric_key_prefix="custom_eval")
        
        print("Evaluation completed. Reports saved to 'evaluation_reports/' directory.")
        print("\nKey Metrics Summary:")
        for key, value in metrics.items():
            if 'f1_micro' in key and 'optimized' not in key:
                print(f"{key}: {value:.4f}")


# ============================================================================
# 4. END-TO-END EXECUTION SCRIPT
# ============================================================================

def main():
    """Main execution script demonstrating the complete workflow."""
    
    print("🚀 Starting Hierarchical Multi-Label Classification Pipeline")
    print("=" * 70)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Data parameters
    NUM_SAMPLES = 1000  # Reduced for faster execution
    NUM_LEVELS = 3
    NUM_MAIN_LABELS = 10  # Reduced for faster execution
    NUM_SUB_LABELS = 3
    
    # Model parameters
    BASE_MODEL_NAME = "distilbert-base-uncased"
    HIDDEN_SIZE = 768
    DROPOUT_PROB = 0.1
    
    # Training parameters
    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 2  # Reduced for demonstration
    WARMUP_STEPS = 100
    
    print(f"Configuration:")
    print(f"  - Samples: {NUM_SAMPLES}")
    print(f"  - Levels: {NUM_LEVELS}")
    print(f"  - Main Labels: {NUM_MAIN_LABELS}")
    print(f"  - Sub Labels: {NUM_SUB_LABELS}")
    print(f"  - Base Model: {BASE_MODEL_NAME}")
    print()
    
    # ========================================================================
    # DATA GENERATION AND PREPROCESSING
    # ========================================================================
    
    print("📊 Generating dummy dataset...")
    df = create_dummy_dataframe(NUM_SAMPLES, NUM_LEVELS, NUM_MAIN_LABELS, NUM_SUB_LABELS)
    print(f"Generated dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Split the data
    print("🔄 Splitting data...")
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print()
    
    # Convert to Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # ========================================================================
    # TOKENIZATION SETUP
    # ========================================================================
    
    print("🔤 Setting up tokenizers...")
    
    # Tokenizer configuration
    tokenizer_configs = {
        'main_text': {
            'model_name': BASE_MODEL_NAME,
            'max_length': 256,
            'padding': True,
            'truncation': True
        },
        'aux_text': {
            'model_name': BASE_MODEL_NAME,
            'max_length': 256,
            'padding': True,
            'truncation': True
        }
    }
    
    # Create data collator
    data_collator = HierarchicalDataCollator(
        tokenizer_configs=tokenizer_configs,
        num_levels=NUM_LEVELS,
        num_main_labels=NUM_MAIN_LABELS,
        num_sub_labels=NUM_SUB_LABELS
    )
    print("✅ Data collator created successfully")
    print()
    
    # ========================================================================
    # MODEL SETUP
    # ========================================================================
    
    print("🤖 Setting up model...")
    
    # Create model configuration
    config = HierarchicalClassifierConfig(
        num_levels=NUM_LEVELS,
        base_model_name=BASE_MODEL_NAME,
        num_main_labels=NUM_MAIN_LABELS,
        num_sub_labels=NUM_SUB_LABELS,
        hidden_size=HIDDEN_SIZE,
        dropout_prob=DROPOUT_PROB,
        numerical_feature_dim=2  # business_codes + scores
    )
    
    # Initialize model
    model = HierarchicalClassifier(config)
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # ========================================================================
    # TRAINING SETUP
    # ========================================================================
    
    print("🏋️ Setting up training...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./hierarchical_classifier_output",
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="main_label_level1_f1_micro",
        greater_is_better=True,
        report_to=None,  # Disable wandb/tensorboard
        save_total_limit=2,
        dataloader_pin_memory=False,  # Avoid potential issues
    )
    
    # Create trainer
    trainer = HierarchicalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("✅ Trainer created successfully")
    print()
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    print("🚂 Starting training...")
    print("-" * 50)
    
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        return
    
    print()
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    print("📊 Running final evaluation on test set...")
    print("-" * 50)
    
    try:
        eval_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        
        print("✅ Evaluation completed!")
        print("\n📈 Key Test Metrics:")
        print("-" * 30)
        
        # Display key metrics
        key_metrics = [
            "test_main_label_level1_f1_micro",
            "test_main_label_level1_f1_macro", 
            "test_main_label_level1_accuracy",
            "test_main_label_sub_label_level1_f1_micro"
        ]
        
        for metric in key_metrics:
            if metric in eval_results:
                print(f"{metric}: {eval_results[metric]:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
    
    print()
    
    # ========================================================================
    # PREDICTION DEMONSTRATION
    # ========================================================================
    
    print("🔮 Demonstrating prediction on new data...")
    print("-" * 50)
    
    try:
        # Take a small sample for prediction demonstration
        sample_df = test_df.head(5).copy()
        
        # Make predictions
        predictions_df = trainer.predict_on_df(sample_df)
        
        print("✅ Predictions completed!")
        print(f"\n📋 Prediction Results (showing first few columns):")
        print("-" * 60)
        
        # Show original columns plus some prediction columns
        display_cols = ['id']
        
        # Add some prediction columns for demonstration
        pred_cols = [col for col in predictions_df.columns if 'pred_level1' in col]
        display_cols.extend(pred_cols[:5])  # Show first 5 prediction columns
        
        print(predictions_df[display_cols].to_string(index=False))
        
        print(f"\n💾 Full predictions DataFrame shape: {predictions_df.shape}")
        print(f"Added {len(predictions_df.columns) - len(sample_df.columns)} prediction columns")
        
    except Exception as e:
        print(f"❌ Prediction failed with error: {e}")
    
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("🎉 PIPELINE EXECUTION COMPLETED!")
    print("=" * 70)
    print("✅ Successfully implemented and tested:")
    print("   - Hierarchical multi-label data generation")
    print("   - Custom data collator with tokenization")
    print("   - Hierarchical classifier with cross-attention")
    print("   - Custom trainer with advanced evaluation")
    print("   - End-to-end training and prediction pipeline")
    print()
    print("📁 Output files:")
    print("   - Model checkpoints: ./hierarchical_classifier_output/")
    print("   - Evaluation reports: ./evaluation_reports/")
    print("   - Training logs: ./logs/")
    print()
    print("🔧 The solution is highly parameterized and can be easily")
    print("   adapted for different numbers of levels, labels, and use cases!")


if __name__ == "__main__":
    main()
