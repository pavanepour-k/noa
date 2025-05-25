# Chinoa from Noa - It means small Noa model (Korean)

This project fine-tunes Microsoft's Phi-2 model for Korean conversational AI using LoRA (Low-Rank Adaptation).

## Dependencies

### Core Requirements
- Python 3.11+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.0+

### Python Packages
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft accelerate bitsandbytes
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Installation

1. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Linux/Mac
# or
env\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify GPU setup:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data Format

The training data should be in JSONL format with the following structure:
```json
{"text": "User: 안녕하세요\nAI: 안녕하세요! 어떻게 도와드릴까요?"}
{"text": "User: 오늘 날씨가 어때요?\nAI: 오늘은 맑고 좋은 날씨네요!"}
```

## Usage

1. Prepare your data:
   - Place training data in `./train_sampled.jsonl`
   - Place validation data in `./valid_asampled.jsonl`

2. Run training:
```bash
python chinoa.py
```

## Model Configuration

- **Base Model**: microsoft/phi-2 (2.7B parameters)
- **Training Method**: LoRA fine-tuning
- **LoRA Config**:
  - Rank (r): 8
  - Alpha: 16
  - Target modules: q_proj, v_proj, k_proj, dense
  - Dropout: 0.05

## Training Parameters

- Batch size: 4 per device
- Learning rate: 2e-4
- Epochs: 2
- Max sequence length: 512
- FP16 precision: Enabled
- Gradient checkpointing: Enabled

## Output

The trained model will be saved to `/mnt/model/chinoa/chinoa_01/`

## Troubleshooting

### CUDA Issues
If you encounter CUDA-related errors:
```bash
# Check CUDA version
nvcc --version

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall bitsandbytes if quantization fails
pip uninstall bitsandbytes -y
pip install bitsandbytes
```

### Memory Issues
If you run out of GPU memory:
- Reduce batch size to 2 or 1
- Enable gradient checkpointing
- Remove quantization if not needed
- Use CPU training (add `no_cuda=True` to TrainingArguments)

## Hardware Requirements

### Minimum
- GPU: 4GB VRAM (e.g. GTX 1050 mobile)
- RAM: 16GB
- Storage: 10GB free space

### Recommended
- GPU: 16GB+ VRAM (e.g., RTX 4080/4090)
- RAM: 32GB+
- Storage: 50GB+ free space