# LLM Fine-Tuning Studio

A production-style Generative AI project that fine-tunes an open-source large language model with LoRA or QLoRA, evaluates the adapted model, and serves it through a FastAPI application.

This project is designed to demonstrate practical skills in:

- Generative AI and large language models
- Supervised fine-tuning
- LoRA and QLoRA parameter-efficient fine-tuning
- Prompt formatting and instruction datasets
- Model evaluation and experiment tracking
- Hugging Face Transformers, PEFT, TRL, and Datasets
- FastAPI inference services
- Docker, testing, configuration management, and CI

## Project Use Case

The example task fine-tunes a language model to act as a software-engineering assistant. The model learns to answer programming questions using clear explanations, structured debugging steps, and safe code suggestions.

The included sample dataset is intentionally small so the repository is easy to understand. For a real experiment, replace it with a larger domain dataset.

## Architecture

```text
Raw JSONL data
      |
      v
Validation and prompt formatting
      |
      v
LoRA or QLoRA supervised fine-tuning
      |
      v
Adapter checkpoint + tokenizer
      |
      +-------------------+
      |                   |
      v                   v
Evaluation pipeline    FastAPI inference service
```

## Main Features

1. **Dataset validation**
   - Checks required fields
   - Removes empty and duplicate examples
   - Produces train and validation splits

2. **Prompt construction**
   - Uses instruction, context, and response fields
   - Supports chat-template formatting when available

3. **Fine-tuning**
   - LoRA or 4-bit QLoRA
   - Gradient accumulation
   - Mixed precision
   - Gradient checkpointing
   - Configurable target modules
   - Checkpoint saving and experiment metadata

4. **Evaluation**
   - Validation loss and perplexity
   - Exact-match and token-overlap metrics
   - Before-and-after generation comparison
   - JSON report generation

5. **Serving**
   - FastAPI REST API
   - Health endpoint
   - Text-generation endpoint
   - Configurable model and adapter paths

6. **Engineering quality**
   - Type hints
   - Logging
   - Unit tests
   - Docker
   - GitHub Actions
   - Environment-based settings

## Repository Structure

```text
llm-finetuning-studio/
├── app/
│   ├── main.py
│   ├── model_service.py
│   └── schemas.py
├── configs/
│   ├── train_cpu_demo.yaml
│   └── train_qlora.yaml
├── data/
│   ├── raw/
│   │   └── sample_instructions.jsonl
│   └── processed/
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── chat.py
├── src/
│   └── llm_finetuning/
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── modeling.py
│       └── prompts.py
├── tests/
│   ├── test_data.py
│   └── test_prompts.py
├── .github/workflows/ci.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Prepare the dataset

```bash
python scripts/prepare_data.py \
  --input data/raw/sample_instructions.jsonl \
  --output-dir data/processed
```

### 4. Run the lightweight demo configuration

This configuration uses a very small model and avoids 4-bit quantization. It is intended for code validation, not model quality.

```bash
python scripts/train.py --config configs/train_cpu_demo.yaml
```

### 5. Run QLoRA fine-tuning on a CUDA GPU

```bash
python scripts/train.py --config configs/train_qlora.yaml
```

The default QLoRA configuration uses:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

You can replace it with another Hugging Face causal language model.

### 6. Evaluate the adapter

```bash
python scripts/evaluate.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path outputs/qwen-software-assistant \
  --data-path data/processed/validation.jsonl \
  --output-path outputs/evaluation_report.json
```

### 7. Run an interactive chat

```bash
python scripts/chat.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path outputs/qwen-software-assistant
```

### 8. Start the API

```bash
export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export ADAPTER_PATH=outputs/qwen-software-assistant
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Explain why this Python function fails.",
    "context": "def divide(a, b): return a / b",
    "max_new_tokens": 150,
    "temperature": 0.2
  }'
```

## Configuration

Training is controlled through YAML files.

Important fields:

| Field | Purpose |
|---|---|
| `model_name` | Hugging Face base model |
| `train_file` | Processed training dataset |
| `validation_file` | Processed validation dataset |
| `use_4bit` | Enables QLoRA-style 4-bit loading |
| `lora_r` | LoRA rank |
| `lora_alpha` | LoRA scaling |
| `target_modules` | Transformer modules adapted by LoRA |
| `max_seq_length` | Maximum tokenized sequence length |
| `learning_rate` | Optimizer learning rate |
| `gradient_accumulation_steps` | Simulates a larger batch size |
| `num_train_epochs` | Number of training passes |
| `output_dir` | Adapter and checkpoint output location |

## Dataset Format

Each line in the source JSONL file must contain:

```json
{
  "instruction": "Explain a technical task.",
  "context": "Optional supporting information.",
  "response": "The desired model answer."
}
```

## Fine-Tuning Strategy

The project uses parameter-efficient fine-tuning instead of updating every model weight.

### LoRA

LoRA inserts small trainable matrices into selected transformer layers. This reduces GPU memory usage and makes checkpoints much smaller.

### QLoRA

QLoRA combines LoRA with a quantized 4-bit base model. The base weights remain frozen while LoRA adapters are trained.

Typical advantages:

- Lower memory use
- Faster experiments
- Smaller deployable adapter files
- Ability to adapt larger models on limited hardware

## Evaluation

The evaluation script produces:

- Average validation loss
- Perplexity
- Exact-match score
- Token-overlap F1
- Generated samples
- Reference answers
- Model configuration metadata

These metrics are intentionally simple and transparent. A production system could add:

- ROUGE or BERTScore
- LLM-as-a-judge evaluation
- Safety evaluation
- Hallucination checks
- Human preference studies
- Domain-specific test suites
- Latency and throughput benchmarks

## Docker

Build:

```bash
docker build -t llm-finetuning-studio .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -e BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  -e ADAPTER_PATH=/models/adapter \
  -v "$(pwd)/outputs/qwen-software-assistant:/models/adapter" \
  llm-finetuning-studio
```

## Testing

```bash
pytest
```



## Important Notes

- QLoRA requires a compatible NVIDIA GPU and `bitsandbytes`.
- The sample dataset is for demonstration only.
- Always review model licenses before commercial use.
- Never train on confidential or copyrighted data without permission.
- Evaluate safety, bias, and hallucination risks before deployment.
