# Lightning Hydra Project Template

## Requirements

Using pip

```bash
pip install -r requirements.txt
```

Using UV

```bash
uv venv
uv pip install -r pyproject.toml
source .venv/bin/activate
```

## Train

```bash
python src/train.py
```

## Tensorboard

```bash
tensorboard --logdir logs
```

## Infer

```bash
python src/infer.py --input_folder samples --output_folder predictions --ckpt_path "/workspace/lightning-template-hydra/logs/catdog_classification/version_5/checkpoints/epoch=0-step=3.ckpt"
```

## Black

```bash
black .
```
