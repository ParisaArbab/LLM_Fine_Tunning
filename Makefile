install:
	pip install -r requirements.txt
	pip install -e .

prepare:
	python scripts/prepare_data.py --input data/raw/sample_instructions.jsonl --output-dir data/processed

train-demo:
	python scripts/train.py --config configs/train_cpu_demo.yaml

train-qlora:
	python scripts/train.py --config configs/train_qlora.yaml

test:
	pytest -q

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

lint:
	python -m compileall src app scripts
