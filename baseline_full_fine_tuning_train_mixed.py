import sys
import transformers, datasets, torch, evaluate, functools, pickle, multiprocessing
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

def tokenize_dataset(examples, *, tokenizer, text_attribute, summary_attribute):
        inputs = ["summarize: " + doc for doc in examples[text_attribute]]
        model_inputs = tokenizer(
            inputs,
            max_length=1024,
            truncation=transformers.tokenization_utils_base.TruncationStrategy.LONGEST_FIRST,
            padding=transformers.utils.PaddingStrategy.LONGEST,
            is_split_into_words=False,
            return_tensors=transformers.utils.TensorType.PYTORCH,
            return_attention_mask=True
        )
        model_inputs['labels'] = tokenizer(
            examples[summary_attribute],
            max_length=256,
            truncation=transformers.tokenization_utils_base.TruncationStrategy.LONGEST_FIRST,
            padding=transformers.utils.PaddingStrategy.LONGEST,
            is_split_into_words=False,
            return_tensors=transformers.utils.TensorType.PYTORCH,
            return_attention_mask=True
        )['input_ids']
        return model_inputs

def main():
    dataset_map = {
        'samsun': {
            'data': datasets.load_dataset('Samsung/samsum'),
            'attributes': {
                'text': 'dialogue',
                'summary': 'summary'
            }
        },
        'cnn': {
            'data': datasets.load_dataset('abisee/cnn_dailymail', '3.0.0'),
            'attributes': {
                'text': 'article',
                'summary': 'highlights'
            }
        }
    }
    bertscore = evaluate.load('bertscore')

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
    output_device = torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    model_run_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = transformers.T5ForConditionalGeneration.from_pretrained('t5-base').to(model_run_device)
    # model_run_device = torch.device('cpu')
    data = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({ k: dataset_map['cnn']['data']['train'][1000:1500][dataset_map['cnn']['attributes'][k]] + dataset_map['samsun']['data']['train'][500:1000][dataset_map['samsun']['attributes'][k]] for k in ['text', 'summary'] }),
        'test': datasets.Dataset.from_dict({ k: dataset_map['cnn']['data']['test'][100:150][dataset_map['cnn']['attributes'][k]] + dataset_map['samsun']['data']['test'][50:100][dataset_map['samsun']['attributes'][k]] for k in ['text', 'summary'] }),
    })
    tokens = data.map(
        functools.partial(tokenize_dataset, tokenizer=tokenizer, text_attribute='text', summary_attribute='summary'),
        batched=True,
        num_proc=multiprocessing.cpu_count() 
    )
    train_len = max([len(x) for x in tokens['train']['input_ids']])
    train_label_len = max([len(x) for x in tokens['train']['labels']])
    test_len = max([len(x) for x in tokens['train']['input_ids']])
    test_label_len = max([len(x) for x in tokens['train']['labels']])
    tokens = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({
            'input_ids': torch.LongTensor([x + [0] * (train_len - len(x)) for x in tokens['train']['input_ids']]),
            'attention_mask': torch.LongTensor([x + [0] * (train_len - len(x)) for x in tokens['train']['attention_mask']]),
            'labels': torch.LongTensor([x + [0] * (train_label_len - len(x)) for x in tokens['train']['labels']]),
        }),
        'test': datasets.Dataset.from_dict({
            'input_ids': torch.LongTensor([x + [0] * (test_len - len(x)) for x in tokens['test']['input_ids']]),
            'attention_mask': torch.LongTensor([x + [0] * (test_len - len(x)) for x in tokens['test']['attention_mask']]),
            'labels': torch.LongTensor([x + [0] * (test_label_len - len(x)) for x in tokens['test']['labels']]),
        })
    })
    training_args = transformers.TrainingArguments(
        output_dir=f'./full-fine-tuning-mixed-t5-base',
        eval_strategy='no',
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=1,
        overwrite_output_dir=True,
        save_strategy='epoch',
        disable_tqdm=False,
        use_cpu=False,
        log_level='info'
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokens['train'],
        eval_dataset=tokens['test']
    )
    trainer.train()
    with open(f'model-t5-base-full-fine-tuning-mixed.pickle', 'wb') as file:
        pickle.dump(trainer.model.to(output_device), file)

    return 0

if __name__ == '__main__':
    sys.exit(main())