# rug-llm-t5-project

## Installation and usage

Create a virtual environment with `pipenv` in the project directory:

```
PIPENV_VENV_IN_PROJECT=1 pipenv install
```

This will also install all necessary dependencies.

Activate the virtual environment:

```
source .venv/bin/activate
```

### Dataset downloading and tokenization

To download and tokenize the datasets execute:

```
python --model t5-base datasets.pickle
```

This will store the tokenized dataset in python loadable file.

### Full-fine tuning

FFT on the datasets.

```
python baseline_full_fine_tuning_train.py
```

FFT on mixed dataset.

```
python baseline_full_fine_tuning_train_mixed.py
```

Generating summaries for evaluation of the FFT (for all datasets):

```
python baseline_full_fine_tuning_train_mixed.py
```

### LoRA

Fine-tune a model using LoRA:

```
python lora_train.py
```

Generate summaries for evaluation using the fine-tune LoRA model:

```
python lora_eval.py
```

### Score

Compute BERT score and generate box plot for the FFT

```
python bertscore_eval.py
```

Compute BERT score and generate box plot for the LoRA PEFT

```
python bertscore_eval_lora.py
```

Compute ROUGE score and generate box plot for FFT

```
python rouge_eval.py
```

Compute ROUGE score and generate box plot for LoRA PEFT

```
python rouge_eval_lora.py
```

