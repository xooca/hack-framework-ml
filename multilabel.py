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
# Creates a sample dataframe that mirrors your described structure.
def get_dummy_data(num_samples: int = 100, num_categories: int = 5) -> pd.DataFrame:
    """Generates a dummy pandas DataFrame for the classification task."""
    print("✅ Generating dummy data...")
    # Long text to ensure tokenizer striding occurs
    long_text_sample = "machine learning " * 200  # Approx. 400 tokens
    
    data = []
    for i in range(num_samples):
        cat_id = f"category_{i % num_categories}"
        main_info = f"This is the main info for id {i}. "
        # Make some main_info fields long to test striding
        if i % 10 == 0:
            main_info += long_text_sample
        else:
            main_info += "It's a standard length entry."

        aux_info = f"Some auxiliary details for {cat_id}."
        label = i % 2 # Simple alternating labels for balance
        
        data.append({
            "id": f"id_{i}",
            "id_aux": cat_id,
            "main_info": main_info,
            "aux_info": aux_info,
            "labels": label
        })
    return pd.DataFrame(data)

# --- 2. Custom Data Structures ---
@dataclass
class AggregatedEvalPrediction(EvalPrediction):
    """
    A custom EvalPrediction class to hold aggregated predictions, labels,
    and the corresponding auxiliary IDs needed for metric calculation.
    """
    id_aux_list: List[str]

# --- 3. Custom Metrics Computer ---
class MetricsComputer:
    """
    A stateful class to compute, manage, and report advanced metrics.
    It finds optimal thresholds per category and generates epoch-level reports.
    """
    def __init__(self):
        self.trainer: Optional[Trainer] = None
        self.thresholds: Dict[str, float] = {}
        print("✅ MetricsComputer initialized.")

    def set_trainer(self, trainer: Trainer):
        """Sets the trainer instance to access its state (e.g., epoch, output_dir)."""
        self.trainer = trainer

    def _calculate_metrics_at_threshold(self, labels, preds_proba, threshold):
        """Helper to calculate a standard set of metrics for a given threshold."""
        preds_binary = (preds_proba > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds_binary, labels=[0, 1]).ravel()
        return {
            "accuracy": accuracy_score(labels, preds_binary),
            "f1": f1_score(labels, preds_binary, zero_division=0),
            "precision": precision_score(labels, preds_binary, zero_division=0),
            "recall": recall_score(labels, preds_binary, zero_division=0),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    def __call__(self, p: AggregatedEvalPrediction) -> Dict[str, float]:
        """
        This method is the core of the metric computation.
        It's called by the Trainer's evaluation loop.
        """
        if self.trainer is None:
            raise ValueError("Trainer must be set using set_trainer() before metric computation.")

        # Unpack the custom prediction object
        logits, labels, id_aux_list = p.predictions, p.label_ids, p.id_aux_list
        
        # Convert logits to probabilities
        preds_proba = torch.sigmoid(torch.from_numpy(logits)).numpy()[:, 1]
        
        df_eval = pd.DataFrame({'probs': preds_proba, 'labels': labels, 'id_aux': id_aux_list})
        
        # --- Point 1: Overall metrics at 0.5 threshold ---
        metrics_at_0_5 = self._calculate_metrics_at_threshold(df_eval['labels'], df_eval['probs'], 0.5)

        # --- Point 2 & 3: Find optimal threshold per category and generate report ---
        per_category_reports = []
        optimized_preds = np.zeros_like(labels)
        
        for category in df_eval['id_aux'].unique():
            cat_df = df_eval[df_eval['id_aux'] == category]
            best_f1 = -1.0
            best_threshold = 0.5
            
            # Find best threshold for this category by maximizing F1
            for threshold in np.linspace(0.01, 0.99, 99):
                f1 = f1_score(cat_df['labels'], (cat_df['probs'] > threshold).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            # Store the optimal threshold
            self.thresholds[category] = best_threshold
            
            # Calculate metrics at the optimized threshold for the report
            optimized_cat_metrics = self._calculate_metrics_at_threshold(
                cat_df['labels'], cat_df['probs'], best_threshold
            )
            
            # Store predictions using the category-specific threshold
            cat_indices = df_eval['id_aux'] == category
            optimized_preds[cat_indices] = (df_eval.loc[cat_indices, 'probs'] > best_threshold).astype(int)

            # --- Point 3: Generate per-category report data ---
            report_row = {
                "category": category,
                "num_samples": len(cat_df),
                "optimal_threshold": best_threshold,
                "f1_optimized": optimized_cat_metrics['f1'],
                "accuracy_optimized": optimized_cat_metrics['accuracy'],
                "precision_optimized": optimized_cat_metrics['precision'],
                "recall_optimized": optimized_cat_metrics['recall'],
            }
            per_category_reports.append(report_row)

        # --- Point 2 (cont.): Calculate overall metrics using optimized thresholds ---
        tn, fp, fn, tp = confusion_matrix(labels, optimized_preds, labels=[0, 1]).ravel()
        metrics_optimized = {
            "accuracy": accuracy_score(labels, optimized_preds),
            "f1": f1_score(labels, optimized_preds, zero_division=0),
            "precision": precision_score(labels, optimized_preds, zero_division=0),
            "recall": recall_score(labels, optimized_preds, zero_division=0),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        
        # --- Point 4: Save the per-category report for the current epoch ---
        epoch = self.trainer.state.epoch or 0
        report_df = pd.DataFrame(per_category_reports).sort_values(by="category").reset_index(drop=True)
        report_path = os.path.join(self.trainer.args.output_dir, f"epoch_{epoch:.2f}_category_report.csv")
        report_df.to_csv(report_path, index=False)
        # print(f"📊 Report for epoch {epoch:.2f} saved to {report_path}")
        
        # --- Point 5: Format final dictionary for logging ---
        final_metrics = {}
        for k, v in metrics_at_0_5.items():
            final_metrics[f"overall_{k}_at_0.5"] = v
        for k, v in metrics_optimized.items():
            final_metrics[f"overall_{k}_optimized"] = v
            
        return final_metrics


# --- 4. Custom Aggregation Trainer ---
class AggregationTrainer(Trainer):
    """
    A custom Trainer that aggregates predictions from strided inputs
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
        """
        Overrides the default evaluate method to perform aggregation before metric calculation.
        """
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # Run the standard prediction loop on the tokenized (and strided) dataset
        # This gives us logits for every single split, not for unique IDs.
        output = self.prediction_loop(
            eval_ds,
            description="Evaluation",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        # --- Aggregation Logic ---
        # Group logits by the original sample index using overflow_to_sample_mapping
        logits_by_id = defaultdict(list)
        for i, logit in enumerate(output.predictions):
            original_idx = eval_ds['overflow_to_sample_mapping'][i]
            logits_by_id[original_idx].append(logit)
            
        # Aggregate the logits and gather corresponding labels and id_aux
        aggregated_logits, aggregated_labels, aggregated_id_aux = [], [], []
        
        unique_original_indices = sorted(logits_by_id.keys())
        for idx in unique_original_indices:
            logits_list = np.array(logits_by_id[idx])
            
            # Apply aggregation strategy
            if self.aggregation_strategy == 'mean':
                agg_logit = np.mean(logits_list, axis=0)
            else: # 'max'
                # For max, we take the max probability for the positive class (logit index 1)
                # This is a common strategy for max aggregation in binary classification
                max_prob_idx = np.argmax(logits_list[:, 1])
                agg_logit = logits_list[max_prob_idx]

            aggregated_logits.append(agg_logit)
            
            # Find the original label and id_aux for this index.
            # We can find the first tokenized sample that maps to this original index.
            first_match_idx = eval_ds['overflow_to_sample_mapping'].index(idx)
            aggregated_labels.append(eval_ds['labels'][first_match_idx])
            aggregated_id_aux.append(eval_ds['id_aux'][first_match_idx])

        # Create our custom prediction object with aggregated results
        aggregated_preds = AggregatedEvalPrediction(
            predictions=np.array(aggregated_logits),
            label_ids=np.array(aggregated_labels),
            id_aux_list=aggregated_id_aux,
        )
        
        # Call our custom metrics computer with the aggregated data
        metrics = self.compute_metrics(aggregated_preds)
        
        # Add the loss to the metrics
        if output.metrics is not None:
          metrics.update(output.metrics)
        
        # Standard trainer logging and control flow
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        return metrics

# --- 5. Main Execution Block ---
def main():
    """Main function to set up and run the training pipeline."""
    print("🚀 Starting NLP Binary Classification Pipeline 🚀")
    
    # --- Configuration ---
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 256  # Max length for each tokenized chunk
    STRIDE = 64       # Overlap between chunks for long texts
    OUTPUT_DIR = "./results"
    AGGREGATION = "mean" # Can be 'mean' or 'max'

    # --- Data Loading and Preprocessing ---
    df = get_dummy_data(num_samples=200, num_categories=4)
    dataset = Dataset.from_pandas(df)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def preprocess_function(examples):
        # Concatenate text fields for classification
        combined_text = [main + " " + aux for main, aux in zip(examples["main_info"], examples["aux_info"])]
        
        tokenized_inputs = tokenizer(
            combined_text,
            max_length=MAX_LENGTH,
            truncation=True,
            stride=STRIDE,
            return_overflowing_tokens=True,
            padding="max_length",
        )
        
        # The `overflow_to_sample_mapping` is key. It maps each new feature (due to striding)
        # back to the index of the original sample in the input.
        sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
        
        # For each new feature, we need to assign it the correct label and id_aux
        # from the original sample it came from.
        labels = []
        id_aux_list = []
        for i in sample_mapping:
            labels.append(examples["labels"][i])
            id_aux_list.append(examples["id_aux"][i])
            
        tokenized_inputs["labels"] = labels
        tokenized_inputs["id_aux"] = id_aux_list
        tokenized_inputs["overflow_to_sample_mapping"] = sample_mapping
        
        return tokenized_inputs

    # The .map() function processes the data and handles the creation of new rows for strided samples.
    # We remove original text columns as they are no longer needed.
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"\nOriginal dataset size: {len(dataset)}")
    print(f"Tokenized dataset size (after striding): {len(tokenized_dataset)}\n")
    
    # Split into train and eval sets
    split_dataset = tokenized_dataset.train_test_split(test_size=0.25, seed=42)
    train_ds = split_dataset["train"]
    eval_ds = split_dataset["test"]
    
    # --- Model and Trainer Setup ---
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Instantiate our custom metrics computer
    metrics_computer = MetricsComputer()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_overall_f1_optimized",
        greater_is_better=True,
        report_to="none", # Disables wandb/mlflow integration for this example
    )
    
    # Instantiate our custom trainer
    trainer = AggregationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics_computer,
        aggregation_strategy=AGGREGATION,
    )

    # CRITICAL STEP: Give the metrics computer a reference to the trainer
    # so it can access state (epoch) and args (output_dir).
    metrics_computer.set_trainer(trainer)

    # --- Train and Evaluate ---
    print("\n🔥 Starting training...")
    trainer.train()
    
    print("\n🏁 Training finished. Running final evaluation...")
    final_metrics = trainer.evaluate()
    
    print("\n--- Final Evaluation Metrics ---")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print(f"\n✅ Pipeline complete! All reports saved in '{OUTPUT_DIR}'.")
    print("Example report from the last epoch:")
    last_epoch = int(trainer.state.epoch)
    last_report_path = os.path.join(OUTPUT_DIR, f"epoch_{float(last_epoch)}_category_report.csv")
    if os.path.exists(last_report_path):
        print(pd.read_csv(last_report_path).to_string())
    else:
        print(f"Could not find report at {last_report_path}")

if __name__ == "__main__":
    main()
