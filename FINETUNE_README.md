# Fine-Tuning Feature

This directory contains a complete implementation for fine-tuning the Microsoft Phi-2 model on custom Q&A datasets using Google Colab.

## 📁 Files

- **`finetuneCollab.py`** - Main training script (single file, runs on Colab)
- **`finetune.md`** - Complete technical documentation
- **`COLAB_QUICKSTART.md`** - Quick start guide for Google Colab
- **`requirements-finetune.txt`** - Python dependencies
- **`dataset_example.xlsx`** - Sample dataset (20 Q&A pairs)

## 🚀 Quick Start

### For Google Colab (Recommended)

1. **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Enable GPU**: Runtime → Change runtime type → GPU

3. **Install dependencies**:
   ```python
   !pip install -q transformers==4.48.0 datasets==2.16.0 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.3 openpyxl==3.1.2 pandas==2.1.4 matplotlib==3.8.2 seaborn==0.13.0
   ```

4. **Upload files**:
   ```python
   from google.colab import files
   files.upload()  # Upload dataset.xlsx and finetuneCollab.py
   ```

5. **Run training**:
   ```python
   !python finetuneCollab.py
   ```

6. **Download results**:
   ```python
   !zip -r fine_tuned_model.zip ./model ./training_visuals
   files.download('fine_tuned_model.zip')
   ```

**Full instructions**: See [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)

## 📊 Dataset Format

Your `dataset.xlsx` should have 2 columns:

| Question | Answer |
|----------|--------|
| What is machine learning? | Machine learning is... |
| How does X work? | X works by... |

- Minimum: 50 Q&A pairs (recommended: 200+)
- Auto-detects column names
- See `dataset_example.xlsx` for reference

## ⚙️ Configuration

Key hyperparameters (edit in `finetuneCollab.py`):

```python
class Config:
    LEARNING_RATE = 2e-4           # Learning rate
    BATCH_SIZE = 4                 # Batch size per GPU
    GRADIENT_ACCUMULATION_STEPS = 4 # Effective batch: 16
    NUM_EPOCHS = 5                 # Training epochs
    MAX_LENGTH = 512               # Max sequence length
    LORA_R = 16                    # LoRA rank
    LORA_ALPHA = 32                # LoRA scaling
```

## 📈 What Gets Created

After training:

```
./model/                    # Fine-tuned model (LoRA adapter)
├── adapter_model.bin       # Trained weights (~40-100 MB)
├── adapter_config.json     # Configuration
└── tokenizer files         # For inference

./training_visuals/         # Training metrics
├── training_curves.png     # Loss over time
├── loss_comparison.png     # Train vs validation
├── training_metrics.json   # Summary stats
├── training_metrics.csv    # Detailed logs
└── training_config.json    # Hyperparameters used

./checkpoints/              # Training checkpoints
└── checkpoint-*/           # Saved every 50 steps
```

## 🎯 Technical Approach

### Method: LoRA (Low-Rank Adaptation)

- **Parameter-efficient**: Only trains ~1.5% of model weights
- **Memory-efficient**: Fits on free Colab GPU (12-16 GB)
- **Fast**: Trains in 20-30 minutes
- **High-quality**: 95-99% of full fine-tuning performance

### Why Not Use .gguf Directly?

GGUF is a quantized inference format (for llama.cpp). We instead:

1. Download original Phi-2 from HuggingFace
2. Fine-tune with LoRA
3. Save LoRA adapter (portable, small)
4. You can later merge with any Phi-2 base model

### Training Pipeline

```
Excel Dataset → Preprocess → Tokenize → LoRA Setup → Train → Save
                                                        ↓
                                            Generate Visualizations
```

## 📚 Documentation

- **Quick Start**: [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md) - Get started in 5 minutes
- **Full Guide**: [finetune.md](finetune.md) - Complete technical documentation
  - What is fine-tuning?
  - How LoRA works
  - Every hyperparameter explained
  - Troubleshooting guide
  - Using your fine-tuned model

## 🔧 Customization

### Change Training Duration

```python
# Faster (less quality)
Config.NUM_EPOCHS = 3

# Better quality (slower)
Config.NUM_EPOCHS = 10
```

### Reduce Memory Usage

```python
Config.BATCH_SIZE = 2
Config.MAX_LENGTH = 256
```

### Increase Model Capacity

```python
Config.LORA_R = 32        # More parameters
Config.LORA_ALPHA = 64    # Stronger learning signal
```

## 🧪 Testing

Test with the example dataset:

```bash
# Local testing (requires GPU)
python finetuneCollab.py

# Colab testing
!python finetuneCollab.py
```

Expected results with `dataset_example.xlsx` (20 samples):
- Training time: ~5-10 minutes
- Final loss: ~0.6-1.2
- Model size: ~40 MB (adapter only)

## 📦 Requirements

- **GPU**: NVIDIA GPU with 12+ GB VRAM (T4, V100, A100)
- **RAM**: 8+ GB
- **Disk**: 10+ GB free space
- **Python**: 3.9+
- **Dependencies**: See `requirements-finetune.txt`

## 🐛 Common Issues

### "CUDA out of memory"
```python
Config.BATCH_SIZE = 2
Config.MAX_LENGTH = 256
```

### "Module not found"
```bash
pip install -r requirements-finetune.txt
```

### "Dataset not found"
Ensure `dataset.xlsx` is in the same directory as the script.

### Training is slow
- Check GPU is enabled
- Reduce epochs or batch size
- Use a more powerful GPU (V100/A100)

## 🎓 Learning Resources

- **What is fine-tuning?** → See [finetune.md](finetune.md#what-is-fine-tuning)
- **How does LoRA work?** → See [finetune.md](finetune.md#how-lora-works)
- **Hyperparameter guide** → See [finetune.md](finetune.md#hyperparameters-explained)

## 💡 Use Cases

This fine-tuning pipeline is ideal for:

- ✅ Domain-specific Q&A systems
- ✅ Customer support chatbots
- ✅ Educational tutoring systems
- ✅ Technical documentation assistants
- ✅ Specialized knowledge bases

## 📊 Performance

**Benchmarks (T4 GPU, 490 samples):**

| Metric | Value |
|--------|-------|
| Training time | 20-30 min |
| GPU memory | 10-12 GB |
| Adapter size | 40-100 MB |
| Training loss | 0.5-0.8 |
| Validation loss | 0.6-0.9 |

## 🚀 Next Steps

1. Prepare your dataset (Excel with Q&A pairs)
2. Follow [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)
3. Run training
4. Evaluate results
5. Deploy your fine-tuned model

## 🤝 Support

- 📖 Full documentation: [finetune.md](finetune.md)
- 🚀 Quick start: [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)
- 💬 Issues: Open a GitHub issue

## 📄 License

MIT License - See main repository LICENSE file

---

**Ready to fine-tune? Start with [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)!** 🎉
