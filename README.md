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

To download and tokenize the datasets use Preprocessing.ipynb.

### Full-fine tuned baselines

To fully fine tune models on all 3 datasets, use FFTBaselineTraining.ipynb. 

### LoRA tuned baselines

To fine tune LoRA baselines and create an untrained model, use LoraBaselinesTraining.ipynb

### Mixed LoRA

Fine-tune a mixed model using LoRA, using LoraMixedTraining.ipynb

### Evaluation

Evaluate the models using Evaluation.ipynb. This will compute ROUGE and BERT scores. Also 

## Notes

### Pickles and Checkpoints

The code frequently saves its output in pickle format. This is useful if your environment 
is prone to crashing, but also to ensure consistency and eliminate needing to run the same 
code over and over again if you're only changing something downstream. 

However by default the code does not load a lot of checkpoints unless it is necessary. As such,
if your environment crashes during training, modify the ```train()``` function to ```train(resume_from_checkpoint=True)``` 
or load from a pickle as necessary. Some functions also happen to act as loaders, so it may not
be necessary. 

### Memory considerations

Training models and generating summaries can be quite resource intensive. Be sure to edit the
batch size to accomodate your available VRAM

Additionally, certain sections of the code have been broken up into individual code blocks despite
it being blatant code reduplication. This is largely also a memory concern. If you only need to 
regenerate the Lora Mixed summaries for example, you don't want to run all the other models at the same time.
Not loading other models while one trains alleviates the load on the GPU a little bit. 

Some segmentation has also been done programatically. Specifically in the evaluation, some steps
have been broken up into parts as they would otherwise eat up too much VRAM and crash the environment. 

### Directories

The directories provided (preprocessing, predictions, readable-predictions, models, and checkpoints) are
required for the code to function. We did not implement the creation of sub-directories. 
