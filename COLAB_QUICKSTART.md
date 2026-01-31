# Google Colab Quick Start Guide

## 🚀 Fine-tune Phi-2 in 5 Minutes

### Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. **Important**: Enable GPU
   - Click `Runtime` → `Change runtime type`
   - Select `GPU` (T4 or better)
   - Click `Save`

### Step 2: Verify GPU is Available

Create a new cell and run:

```python
import torch
print("🔍 Checking GPU...")
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("✅ GPU is ready!")
else:
    print("❌ GPU not enabled. Please enable it in Runtime → Change runtime type")
```

### Step 3: Install Required Packages

In a new cell:

```python
print("📦 Installing required packages (this takes 2-3 minutes)...")
!pip install -q transformers==4.48.0 datasets==2.16.0 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.3 openpyxl==3.1.2 pandas==2.1.4 matplotlib==3.8.2 seaborn==0.13.0

print("✅ All packages installed!")
```

### Step 4: Upload Your Files

In a new cell:

```python
from google.colab import files

# Upload dataset
print("📤 Upload your dataset.xlsx file:")
uploaded = files.upload()

# Upload training script
print("\n📤 Upload your finetuneCollab.py file:")
uploaded = files.upload()

print("\n✅ Files uploaded successfully!")
print("Files in current directory:")
!ls -lh
```

### Step 5: Run Fine-Tuning

In a new cell:

```python
# Run the fine-tuning script
!python finetuneCollab.py
```

### Step 6: Monitor Training

You'll see output like:
```
================================================================================
  🚀 Microsoft Phi-2 Fine-Tuning Script
================================================================================

✅ GPU Available: Tesla T4
   Memory: 15.90 GB

[Step 1] Loading Dataset
✅ Loaded dataset with 490 rows

[Step 2] Preprocessing Dataset
✅ Preprocessed 490 valid Q&A pairs
✅ Training samples: 441
✅ Validation samples: 49

[Step 3] Loading Model and Tokenizer
🔄 Loading model (this may take a few minutes)...
✅ Model loaded: microsoft/phi-2
   Parameters: 2.78B

[Step 4] Configuring LoRA
✅ LoRA configured
   Trainable parameters: 42,467,328 (1.53%)

[Step 5] Training Model
🚀 Starting training...
...
```

Training takes 20-30 minutes on a T4 GPU.

### Step 7: Download Results

After training completes, download your fine-tuned model:

```python
# Create a zip file with all outputs
!zip -r fine_tuned_model.zip ./model ./training_visuals ./checkpoints

# Download to your computer
from google.colab import files
files.download('fine_tuned_model.zip')

print("✅ Download complete!")
```

### Step 8: View Training Results

To see training curves in Colab:

```python
from IPython.display import Image, display
import os

# Display training curves
if os.path.exists('./training_visuals/training_curves.png'):
    display(Image('./training_visuals/training_curves.png'))
    
if os.path.exists('./training_visuals/loss_comparison.png'):
    display(Image('./training_visuals/loss_comparison.png'))

# Show training metrics
import json
with open('./training_visuals/training_metrics.json', 'r') as f:
    metrics = json.load(f)
    print("\n📊 Training Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")
```

---

## 🎯 Complete Single-Cell Script

For convenience, here's everything in one cell (except file uploads):

```python
# Complete Fine-Tuning Script for Google Colab
# Run each section in order

# 1. Check GPU
import torch
print("✅ GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# 2. Install packages (if not already installed)
try:
    import transformers
    print("✅ Packages already installed")
except:
    print("📦 Installing packages...")
    !pip install -q transformers==4.48.0 datasets==2.16.0 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.3 openpyxl==3.1.2 pandas==2.1.4 matplotlib==3.8.2 seaborn==0.13.0

# 3. Upload files (uncomment to upload)
# from google.colab import files
# print("Upload dataset.xlsx:"); files.upload()
# print("Upload finetuneCollab.py:"); files.upload()

# 4. Run training
!python finetuneCollab.py

# 5. Display results
from IPython.display import Image, display
display(Image('./training_visuals/training_curves.png'))

# 6. Download model (uncomment to download)
# !zip -r fine_tuned_model.zip ./model ./training_visuals
# files.download('fine_tuned_model.zip')
```

---

## 📝 Dataset Format

Your `dataset.xlsx` should look like this:

| Question | Answer |
|----------|--------|
| What is machine learning? | Machine learning is a subset of AI that enables computers to learn from data... |
| How does a neural network work? | A neural network is composed of layers of interconnected nodes... |
| ... | ... |

**Requirements:**
- Excel file (.xlsx format)
- At least 2 columns
- Column names can be anything (auto-detected)
- Minimum 50 Q&A pairs (recommended: 200+)
- No empty rows

---

## ⚙️ Customization

To change training parameters, edit these values in `finetuneCollab.py`:

```python
class Config:
    # Current defaults optimized for Tesla T4 GPU (14.74 GB)
    BATCH_SIZE = 1  # Optimized for memory efficiency
    GRADIENT_ACCUMULATION_STEPS = 16  # Maintains effective batch size of 16
    MAX_LENGTH = 256  # Reduced to fit in GPU memory
    NUM_EPOCHS = 5
    
    # --- Alternative configurations (choose one) ---
    
    # Option A: Fast training (less quality)
    # NUM_EPOCHS = 3
    
    # Option B: Better quality (slower training)
    # NUM_EPOCHS = 10
    # LORA_R = 32
    
    # Option C: Further reduce memory usage if still encountering OOM
    # MAX_LENGTH = 128
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"
The script now includes memory optimizations by default:
- Batch size: 1 (reduced from 4)
- Max length: 256 (reduced from 512)
- Gradient checkpointing: enabled

If you still encounter memory issues, you can further reduce MAX_LENGTH to 128.

### "No module named 'transformers'"
```python
# Run installation again:
!pip install --upgrade transformers==4.48.0 datasets peft accelerate bitsandbytes
```

### "Cannot find dataset.xlsx"
```python
# Check files in current directory:
!ls -la

# Make sure dataset.xlsx is in the root directory
# Re-upload if necessary
```

### Training is slow
- Verify GPU is enabled (Runtime → Change runtime type)
- Try reducing epochs to 3
- Check GPU usage: `!nvidia-smi`

### Model download fails
```python
# Login to HuggingFace (free account needed)
from huggingface_hub import login
login(token="your_token_here")  # Get from huggingface.co/settings/tokens
```

---

## 🎓 What Happens During Training?

1. **Downloads Phi-2** from HuggingFace (~5 GB, takes 5-10 min)
2. **Loads your dataset** from Excel
3. **Applies LoRA** adapters (makes training efficient)
4. **Trains for 5 epochs** (~4 minutes per epoch on T4)
5. **Saves checkpoints** every 50 steps
6. **Generates visualizations** of training progress
7. **Saves fine-tuned model** to `./model` directory

**Total time:** 30-40 minutes on T4 GPU

---

## 💡 Tips

1. **Start small**: Test with 50-100 samples first
2. **Monitor loss**: Should decrease from ~2-3 to ~0.5-1.0
3. **Check visuals**: Training curves show if model is learning
4. **Save often**: Colab can disconnect, but checkpoints are saved
5. **Use GPU**: Training on CPU is 50x slower

---

## 🚀 Next Steps After Training

1. **Test the model** on new questions (not in training data)
2. **Compare with base model** to see improvement
3. **Iterate**: Adjust hyperparameters if needed
4. **Deploy**: Use the model in your application

See `finetune.md` for detailed information on using your fine-tuned model!

---

## 📚 Files Created

After successful training:

```
/content/
├── dataset.xlsx              # Your input
├── finetuneCollab.py         # Training script
├── model/                    # ✨ Fine-tuned model (download this!)
│   ├── adapter_model.bin     # LoRA weights
│   ├── adapter_config.json   # Configuration
│   └── tokenizer files       # For inference
├── checkpoints/              # Training checkpoints
└── training_visuals/         # 📊 Metrics and graphs
    ├── training_curves.png
    ├── loss_comparison.png
    └── training_metrics.json
```

---

**Questions?** Check the full documentation in `finetune.md`

**Happy Fine-Tuning! 🎉**
