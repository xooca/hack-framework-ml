import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import defaultdict

# Suppress verbose logging from transformers
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
import logging
logging.disable(logging.WARNING)


# --- 1. Dummy Data Generation ---
def get_dummy_data(num_samples: int = 100, num_categories: int = 5) -> pd.DataFrame:
    """Generates a dummy pandas DataFrame for the classification task."""
    print("✅ Generating dummy data...")
    long_main_info_sample = "main info machine learning model training pipeline " * 150
    long_aux_info_sample = "auxiliary metadata point " * 50
    
    data = []
    for i in range(num_samples):
        cat_id = f"category_{i % num_categories}"
        if i % 10 == 0:
            main_info = f"This is a long main info for id {i}. {long_main_info_sample}"
            aux_info = f"This is a shorter aux info for id {i}. {long_aux_info_sample}"
        else:
            main_info = f"Standard length main info for id {i}."
            aux_info = f"Standard length aux info for {cat_id}."
            
        label = i % 2
        
        data.append({
            "id": f"unique_id_{i}", # Explicit unique ID
            "id_aux": cat_id,
            "main_info": main_info,
            "aux_info": aux_info,
            "labels": label
        })
    return pd.DataFrame(data)

# --- 2. Custom Data Structures ---
@dataclass
class AggregatedEvalPrediction(EvalPrediction):
    """A custom EvalPrediction to hold aggregated results and id_aux for metrics."""
    id_aux_list: List[str]

# --- 3. Custom Metrics Computer ---
class MetricsComputer:
    """A stateful class to compute, manage, and report advanced metrics."""
    def __init__(self):
        self.trainer: Optional[Trainer] = None
        self.thresholds: Dict[str, float] = {}
        print("✅ MetricsComputer initialized.")

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer

    def _calculate_metrics_at_threshold(self, labels, preds_proba, threshold):
        preds_binary = (preds_proba > threshold).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(labels, preds_binary, labels=[0, 1]).ravel()
        except ValueError:
            tn, fp, fn, tp = 0, 0, 0, 0
            if len(labels) > 0:
                if all(preds_binary == 0): tn = len(labels)
                elif all(preds_binary == 1): fp = len(labels)
        return {
            "accuracy": accuracy_score(labels, preds_binary),
            "f1": f1_score(labels, preds_binary, zero_division=0),
            "precision": precision_score(labels, preds_binary, zero_division=0),
            "recall": recall_score(labels, preds_binary, zero_division=0),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    def __call__(self, p: AggregatedEvalPrediction) -> Dict[str, float]:
        if self.trainer is None:
            raise ValueError("Trainer must be set using set_trainer() before metric computation.")

        logits, labels, id_aux_list = p.predictions, p.label_ids, p.id_aux_list
        preds_proba = torch.sigmoid(torch.from_numpy(logits)).numpy()[:, 1]
        
        df_eval = pd.DataFrame({'probs': preds_proba, 'labels': labels, 'id_aux': id_aux_list})
        
        metrics_at_0_5 = self._calculate_metrics_at_threshold(df_eval['labels'], df_eval['probs'], 0.5)

        per_category_reports, optimized_preds = [], np.zeros_like(labels)
        
        for category in df_eval['id_aux'].unique():
            cat_df = df_eval[df_eval['id_aux'] == category]
            best_f1, best_threshold = -1.0, 0.5
            
            for threshold in np.linspace(0.01, 0.99, 99):
                f1 = f1_score(cat_df['labels'], (cat_df['probs'] > threshold).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_threshold = f1, threshold
            
            self.thresholds[category] = best_threshold
            optimized_cat_metrics = self._calculate_metrics_at_threshold(cat_df['labels'], cat_df['probs'], best_threshold)
            
            cat_indices = df_eval['id_aux'] == category
            optimized_preds[cat_indices] = (df_eval.loc[cat_indices, 'probs'] > best_threshold).astype(int)

            per_category_reports.append({
                "category": category, "num_samples": len(cat_df), "optimal_threshold": best_threshold,
                "f1_optimized": optimized_cat_metrics['f1'], "accuracy_optimized": optimized_cat_metrics['accuracy'],
                "precision_optimized": optimized_cat_metrics['precision'], "recall_optimized": optimized_cat_metrics['recall'],
            })

        tn, fp, fn, tp = confusion_matrix(labels, optimized_preds, labels=[0, 1]).ravel()
        metrics_optimized = {
            "accuracy": accuracy_score(labels, optimized_preds), "f1": f1_score(labels, optimized_preds, zero_division=0),
            "precision": precision_score(labels, optimized_preds, zero_division=0), "recall": recall_score(labels, optimized_preds, zero_division=0),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        
        epoch = self.trainer.state.epoch or 0
        report_df = pd.DataFrame(per_category_reports).sort_values(by="category").reset_index(drop=True)
        report_path = os.path.join(self.trainer.args.output_dir, f"epoch_{epoch:.2f}_category_report.csv")
        report_df.to_csv(report_path, index=False)
        
        final_metrics = {}
        for k, v in metrics_at_0_5.items(): final_metrics[f"overall_{k}_at_0.5"] = v
        for k, v in metrics_optimized.items(): final_metrics[f"overall_{k}_optimized"] = v
        return final_metrics

# --- 4. Custom Aggregation Trainer ---
class AggregationTrainer(Trainer):
    """
    A custom Trainer that aggregates predictions from strided inputs based on `id`
    before computing metrics.
    """
    def __init__(self, *args, aggregation_strategy='mean', **kwargs):
        super().__init__(*args, **kwargs)
        if aggregation_strategy not in ['mean', 'max']:
            raise ValueError("aggregation_strategy must be 'mean' or 'max'")
        self.aggregation_strategy = aggregation_strategy
        print(f"✅ AggregationTrainer initialized with strategy: '{self.aggregation_strategy}'.")

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # This gives us logits for every single split, not for unique IDs.
        output = self.prediction_loop(eval_ds, description="Evaluation", prediction_loss_only=False,
                                      ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
        # --- NEW AGGREGATION LOGIC (using 'id') ---
        # Group logits and metadata explicitly by the 'id' column.
        logits_by_id = defaultdict(list)
        metadata_by_id = {} # To store label and id_aux for each unique id

        for i, logit in enumerate(output.predictions):
            # Use the 'id' column from the evaluation dataset for grouping
            sample_id = eval_ds['id'][i]
            logits_by_id[sample_id].append(logit)
            
            # Store metadata only once per id to avoid redundancy
            if sample_id not in metadata_by_id:
                metadata_by_id[sample_id] = {
                    'label': eval_ds['labels'][i],
                    'id_aux': eval_ds['id_aux'][i]
                }
        
        aggregated_logits, aggregated_labels, aggregated_id_aux = [], [], []
        
        # Sort keys for deterministic order
        for sample_id in sorted(logits_by_id.keys()):
            logits_list = np.array(logits_by_id[sample_id])
            
            if self.aggregation_strategy == 'mean':
                agg_logit = np.mean(logits_list, axis=0)
            else: # 'max' strategy
                max_prob_idx = np.argmax(logits_list[:, 1])
                agg_logit = logits_list[max_prob_idx]

            aggregated_logits.append(agg_logit)
            aggregated_labels.append(metadata_by_id[sample_id]['label'])
            aggregated_id_aux.append(metadata_by_id[sample_id]['id_aux'])

        # Create our custom prediction object with aggregated results
        aggregated_preds = AggregatedEvalPrediction(
            predictions=np.array(aggregated_logits),
            label_ids=np.array(aggregated_labels),
            id_aux_list=aggregated_id_aux,
        )
        
        # Call metrics computer with the correctly aggregated data
        metrics = self.compute_metrics(aggregated_preds)
        
        if output.metrics is not None:
          metrics.update(output.metrics)
        
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

# --- 5. Main Execution Block ---
def main():
    print("🚀 Starting NLP Binary Classification Pipeline 🚀")
    
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH, STRIDE = 256, 64
    OUTPUT_DIR, AGGREGATION = "./results_id_aggregation", "mean"

    df = get_dummy_data(num_samples=200, num_categories=4)
    dataset = Dataset.from_pandas(df)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # --- UPDATED PREPROCESS FUNCTION (now includes 'id') ---
    def preprocess_function(examples):
        tokenized_inputs = tokenizer(
            text=examples["main_info"], text_pair=examples["aux_info"],
            max_length=MAX_LENGTH, truncation='longest_first', stride=STRIDE,
            return_overflowing_tokens=True, padding="max_length",
        )
        
        sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
        
        # Map id, id_aux, and labels to each strided segment
        id_list, id_aux_list, labels = [], [], []
        for i in sample_mapping:
            id_list.append(examples["id"][i]) # Carry the 'id' forward
            id_aux_list.append(examples["id_aux"][i])
            labels.append(examples["labels"][i])
            
        tokenized_inputs["id"] = id_list
        tokenized_inputs["id_aux"] = id_aux_list
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("\n✅ Preprocessing data: carrying 'id' and 'id_aux' to strided samples...")
    # 'id' is now also a column in the tokenized dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Tokenized dataset size (after striding): {len(tokenized_dataset)}\n")
    
    split_dataset = tokenized_dataset.train_test_split(test_size=0.25, seed=42)
    train_ds, eval_ds = split_dataset["train"], split_dataset["test"]
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    metrics_computer = MetricsComputer()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=3,
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        evaluation_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_overall_f1_optimized",
        greater_is_better=True, report_to="none",
    )
    
    trainer = AggregationTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=tokenizer, compute_metrics=metrics_computer, aggregation_strategy=AGGREGATION,
    )
    metrics_computer.set_trainer(trainer)

    print("\n🔥 Starting training...")
    trainer.train()
    
    print("\n🏁 Training finished. Running final evaluation...")
    final_metrics = trainer.evaluate()
    
    print("\n--- Final Evaluation Metrics ---")
    for key, value in final_metrics.items(): print(f"{key}: {value:.4f}")
    
    print(f"\n✅ Pipeline complete! All reports saved in '{OUTPUT_DIR}'.")
    last_epoch = int(training_args.num_train_epochs)
    last_report_path = os.path.join(OUTPUT_DIR, f"epoch_{float(last_epoch)}_category_report.csv")
    if os.path.exists(last_report_path):
        print("Example report from the last epoch:")
        print(pd.read_csv(last_report_path).to_string())

if __name__ == "__main__":
    main()
