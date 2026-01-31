"""
Fine-tuning Script for Microsoft Phi-2 Model on Google Colab
==============================================================

This script fine-tunes the Microsoft Phi-2 model on a custom Q&A dataset.
Designed to run on Google Colab with minimal setup.

Usage:
1. Upload your model files and dataset.xlsx to the main Colab directory
2. Run this script: python finetuneCollab.py
3. Fine-tuned model will be saved to ./model directory

Author: Fine-tune LLM Team
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Check if running on Colab
try:
    import google.colab
    IN_COLAB = True
    print("🚀 Running on Google Colab")
except:
    IN_COLAB = False
    print("💻 Running on local environment")

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset as HFDataset

# Set up plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for fine-tuning"""
    
    # Paths
    DATASET_PATH = "dataset.xlsx"
    MODEL_NAME = "microsoft/phi-2"  # We'll download from HuggingFace
    OUTPUT_DIR = "./model"
    CHECKPOINT_DIR = "./checkpoints"
    VISUALS_DIR = "./training_visuals"
    
    # Training hyperparameters
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 1  # Reduced from 4 to 1 to fit in GPU memory
    GRADIENT_ACCUMULATION_STEPS = 16  # Increased from 4 to 16 to maintain effective batch size of 16
    NUM_EPOCHS = 5
    MAX_LENGTH = 256  # Reduced from 512 to 256 to use less memory per sample
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.01
    
    # LoRA parameters
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Other settings
    SEED = 42
    FP16 = torch.cuda.is_available()
    SAVE_STEPS = 50
    LOGGING_STEPS = 10
    EVAL_STEPS = 50
    SAVE_TOTAL_LIMIT = 3

# Create directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.VISUALS_DIR, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_step(step_num, text):
    """Print a formatted step"""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 60)

def check_gpu():
    """Check GPU availability and print info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.2f} GB")
        return True
    else:
        print("⚠️  No GPU available, using CPU (this will be slow)")
        return False

def save_training_config():
    """Save training configuration to JSON"""
    config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('_')}
    config_dict['FP16'] = str(config_dict['FP16'])
    
    with open(os.path.join(Config.VISUALS_DIR, 'training_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✅ Configuration saved to {Config.VISUALS_DIR}/training_config.json")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_dataset(file_path):
    """Load dataset from Excel file"""
    print_step(1, "Loading Dataset")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Dataset file not found: {file_path}")
    
    # Load Excel file
    df = pd.read_excel(file_path)
    print(f"✅ Loaded dataset with {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Display first few rows
    print("\n📊 Sample data:")
    print(df.head(3))
    
    return df

def preprocess_dataset(df, tokenizer):
    """Preprocess dataset for fine-tuning"""
    print_step(2, "Preprocessing Dataset")
    
    # Identify question and answer columns (flexible column detection)
    columns = df.columns.tolist()
    question_col = None
    answer_col = None
    
    # Try to find question and answer columns
    for col in columns:
        col_lower = col.lower()
        if 'question' in col_lower or 'q' == col_lower or 'query' in col_lower:
            question_col = col
        elif 'answer' in col_lower or 'a' == col_lower or 'response' in col_lower:
            answer_col = col
    
    # If not found, use first two columns
    if question_col is None or answer_col is None:
        print("⚠️  Could not auto-detect columns. Using first two columns.")
        question_col = columns[0]
        answer_col = columns[1]
    
    print(f"✅ Using columns: Question='{question_col}', Answer='{answer_col}'")
    
    # Create formatted prompts
    formatted_data = []
    for idx, row in df.iterrows():
        question = str(row[question_col]).strip()
        answer = str(row[answer_col]).strip()
        
        # Skip empty or invalid rows
        if not question or not answer or question == 'nan' or answer == 'nan':
            continue
        
        # Format as instruction-following prompt
        prompt = f"### Question: {question}\n\n### Answer: {answer}"
        formatted_data.append({"text": prompt})
    
    print(f"✅ Preprocessed {len(formatted_data)} valid Q&A pairs")
    
    # Create HuggingFace dataset
    hf_dataset = HFDataset.from_list(formatted_data)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding="max_length",
            return_tensors=None
        )
    
    print("🔄 Tokenizing dataset...")
    tokenized_dataset = hf_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Split into train and validation (90/10 split)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=Config.SEED)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer():
    """Setup model and tokenizer"""
    print_step(3, "Loading Model and Tokenizer")
    
    # Load tokenizer first
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Set pad_token = eos_token")
    
    # Load model configuration and set pad_token_id BEFORE loading the model
    print("🔄 Loading model configuration...")
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    config.pad_token_id = tokenizer.pad_token_id  # Set pad_token_id in config
    
    # Load model with updated configuration
    print("🔄 Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        config=config,  # Pass the updated config
        trust_remote_code=True,
        torch_dtype=torch.float16 if Config.FP16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Enable gradient checkpointing to reduce memory usage
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    print(f"✅ Model loaded: {Config.MODEL_NAME}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"   pad_token_id: {config.pad_token_id}")
    
    return model, tokenizer

def setup_lora(model):
    """Setup LoRA for parameter-efficient fine-tuning"""
    print_step(4, "Configuring LoRA")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=Config.LORA_TARGET_MODULES
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA configured")
    print(f"   Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"   Total parameters: {total_params:,}")
    
    return model

# ============================================================================
# TRAINING
# ============================================================================

class TrainingMetricsCallback:
    """Callback to track and visualize training metrics"""
    
    def __init__(self):
        self.losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.steps = []
        self.eval_steps = []
        
    def on_log(self, logs):
        """Called when logging occurs"""
        if 'loss' in logs:
            self.losses.append(logs['loss'])
            self.steps.append(logs.get('step', len(self.losses)))
        
        if 'learning_rate' in logs:
            self.learning_rates.append(logs['learning_rate'])
        
        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])
            self.eval_steps.append(logs.get('step', len(self.eval_losses)))

def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Train the model"""
    print_step(5, "Training Model")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=Config.CHECKPOINT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=Config.WARMUP_STEPS,
        logging_steps=Config.LOGGING_STEPS,
        save_steps=Config.SAVE_STEPS,
        eval_steps=Config.EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        fp16=Config.FP16,
        report_to="none",
        seed=Config.SEED,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("🚀 Starting training...")
    print(f"   Total epochs: {Config.NUM_EPOCHS}")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    print(f"   Gradient accumulation: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Effective batch size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Learning rate: {Config.LEARNING_RATE}")
    
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time/60:.2f} minutes")
    print(f"   Final training loss: {train_result.training_loss:.4f}")
    
    # Save training metrics
    metrics = {
        'training_loss': float(train_result.training_loss),
        'training_time_minutes': training_time / 60,
        'epochs': Config.NUM_EPOCHS,
        'total_steps': train_result.global_step
    }
    
    with open(os.path.join(Config.VISUALS_DIR, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return trainer, train_result

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_training_visualizations(trainer):
    """Create and save training visualizations"""
    print_step(6, "Creating Visualizations")
    
    # Get training history
    history = trainer.state.log_history
    
    # Extract metrics
    train_losses = []
    eval_losses = []
    learning_rates = []
    steps = []
    eval_steps = []
    
    for entry in history:
        if 'loss' in entry:
            train_losses.append(entry['loss'])
            steps.append(entry['step'])
            if 'learning_rate' in entry:
                learning_rates.append(entry['learning_rate'])
        
        if 'eval_loss' in entry:
            eval_losses.append(entry['eval_loss'])
            eval_steps.append(entry['step'])
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Training Loss
    axes[0, 0].plot(steps, train_losses, 'b-', linewidth=2, label='Training Loss')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Evaluation Loss
    if eval_losses:
        axes[0, 1].plot(eval_steps, eval_losses, 'r-', linewidth=2, label='Validation Loss')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning Rate Schedule
    if learning_rates:
        axes[1, 0].plot(steps[:len(learning_rates)], learning_rates, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 4. Combined Train/Eval Loss
    if eval_losses:
        axes[1, 1].plot(steps, train_losses, 'b-', linewidth=2, label='Training', alpha=0.7)
        axes[1, 1].plot(eval_steps, eval_losses, 'r-', linewidth=2, label='Validation', alpha=0.7)
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training vs Validation Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(Config.VISUALS_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to {save_path}")
    
    # Create loss comparison chart
    if eval_losses:
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate moving average
        window = min(10, len(train_losses) // 10) if len(train_losses) > 10 else 1
        if window > 1:
            train_ma = pd.Series(train_losses).rolling(window=window).mean()
            ax.plot(steps, train_ma, 'b-', linewidth=3, label='Training (MA)', alpha=0.8)
        ax.plot(steps, train_losses, 'b-', linewidth=1, alpha=0.3, label='Training (raw)')
        
        ax.plot(eval_steps, eval_losses, 'ro-', linewidth=2, markersize=8, 
                label='Validation', alpha=0.8)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Model Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Save
        save_path2 = os.path.join(Config.VISUALS_DIR, 'loss_comparison.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"✅ Loss comparison saved to {save_path2}")
    
    plt.close('all')
    
    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'step': steps,
        'train_loss': train_losses,
    })
    
    if learning_rates:
        metrics_df['learning_rate'] = learning_rates + [np.nan] * (len(steps) - len(learning_rates))
    
    csv_path = os.path.join(Config.VISUALS_DIR, 'training_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"✅ Training metrics saved to {csv_path}")

# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model(model, tokenizer):
    """Save the fine-tuned model"""
    print_step(7, "Saving Model")
    
    # Save LoRA adapter
    print("💾 Saving LoRA adapter...")
    model.save_pretrained(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    
    print(f"✅ Model saved to {Config.OUTPUT_DIR}")
    print(f"   Files saved:")
    for file in os.listdir(Config.OUTPUT_DIR):
        print(f"   - {file}")
    
    # Create a README for the model
    readme_content = f"""# Fine-tuned Microsoft Phi-2 Model

## Training Details

- **Base Model**: microsoft/phi-2
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Dataset Size**: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS} effective batch size
- **Epochs**: {Config.NUM_EPOCHS}
- **Learning Rate**: {Config.LEARNING_RATE}

## LoRA Configuration

- **r**: {Config.LORA_R}
- **alpha**: {Config.LORA_ALPHA}
- **dropout**: {Config.LORA_DROPOUT}
- **target_modules**: {', '.join(Config.LORA_TARGET_MODULES)}

## Usage

To load this model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{Config.OUTPUT_DIR}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{Config.OUTPUT_DIR}")

# Generate text
inputs = tokenizer("### Question: Your question here\\n\\n### Answer:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## Files

- `adapter_model.bin`: LoRA adapter weights
- `adapter_config.json`: LoRA configuration
- `tokenizer files`: Tokenizer configuration
"""
    
    with open(os.path.join(Config.OUTPUT_DIR, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Model README saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Print banner
    print_header("🚀 Microsoft Phi-2 Fine-Tuning Script")
    print("This script will fine-tune the Phi-2 model on your custom Q&A dataset")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check environment
    has_gpu = check_gpu()
    
    # Set random seed for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    print(f"✅ Random seed set to {Config.SEED}")
    
    # Save configuration
    save_training_config()
    
    try:
        # Step 1: Load dataset
        df = load_dataset(Config.DATASET_PATH)
        
        # Step 2: Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # Step 3: Preprocess dataset
        train_dataset, eval_dataset = preprocess_dataset(df, tokenizer)
        
        # Step 4: Setup LoRA
        model = setup_lora(model)
        
        # Step 5: Train model
        trainer, train_result = train_model(model, tokenizer, train_dataset, eval_dataset)
        
        # Step 6: Create visualizations
        create_training_visualizations(trainer)
        
        # Step 7: Save model
        save_model(model, tokenizer)
        
        # Success message
        print_header("✅ Fine-Tuning Complete!")
        print("Your fine-tuned model is ready to use!")
        print(f"\n📂 Output locations:")
        print(f"   Model: {Config.OUTPUT_DIR}")
        print(f"   Checkpoints: {Config.CHECKPOINT_DIR}")
        print(f"   Visualizations: {Config.VISUALS_DIR}")
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print_header("❌ Error During Fine-Tuning")
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
