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

    output_device = torch.device('cpu')
    model_run_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model_run_device = torch.device('cpu')

    with open('model-t5-base-lora-fine-tuning-cnn.pickle', 'rb') as file:
        model_cnn = pickle.load(file).to(model_run_device)
    with open('model-t5-base-lora-fine-tuning-samsun.pickle', 'rb') as file:
        model_samsun = pickle.load(file).to(model_run_device)
    
    model = transformers.T5ForConditionalGeneration.from_pretrained('t5-base').to(model_run_device)

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')

    for dataset_name in dataset_map:
        data = datasets.DatasetDict({
            'test': datasets.Dataset.from_dict({ k: dataset_map[dataset_name]['data']['test'][100:200][dataset_map[dataset_name]['attributes'][k]] for k in ['text', 'summary'] }),
        })
        tokens = data.map(
            functools.partial(tokenize_dataset, tokenizer=tokenizer, text_attribute='text', summary_attribute='summary'),
            batched=True,
            num_proc=multiprocessing.cpu_count() 
        )
        test_len = max([len(x) for x in tokens['test']['input_ids']])
        test_label_len = max([len(x) for x in tokens['test']['labels']])
        tokens = datasets.DatasetDict({
            'test': datasets.Dataset.from_dict({
                'summary': tokens['test']['summary'],
                'input_ids': torch.LongTensor([x + [0] * (test_len - len(x)) for x in tokens['test']['input_ids']]),
                'attention_mask': torch.LongTensor([x + [0] * (test_len - len(x)) for x in tokens['test']['attention_mask']]),
                'labels': torch.LongTensor([x + [0] * (test_label_len - len(x)) for x in tokens['test']['labels']]),
            })
        })
        predictions = {
            'references': [],
            'pretrained': [],
            'cnn': [],
            'samsun': []
        }
        for example in tqdm(tokens['test'], desc='Generating summaries'):
            input_ids = torch.LongTensor(example['input_ids'])[None].to(model_run_device)
            pretrained_out_ids = model.generate(input_ids=input_ids, max_length=192, min_length=10, num_beams=4, length_penalty=2.0, use_cache=False).to(output_device)
            cnn_out_ids = model_cnn.generate(input_ids=input_ids, max_length=192, min_length=10, num_beams=4, length_penalty=2.0, use_cache=False).to(output_device)
            samsun_out_ids = model_samsun.generate(input_ids=input_ids, max_length=192, min_length=10, num_beams=4, length_penalty=2.0, use_cache=False).to(output_device)
            pretrained_summary = tokenizer.decode(pretrained_out_ids[0], skip_special_tokens=True)
            cnn_summary = tokenizer.decode(cnn_out_ids[0], skip_special_tokens=True)
            samsun_summary = tokenizer.decode(samsun_out_ids[0], skip_special_tokens=True)
            predictions['references'].append(example['summary'])
            predictions['pretrained'].append(pretrained_summary)
            predictions['cnn'].append(cnn_summary)
            predictions['samsun'].append(samsun_summary)

        with open(f'lora-fine-tuning-predictions-{dataset_name}.pickle', 'wb') as file:
            pickle.dump(predictions, file)

    return 0

if __name__ == '__main__':
    sys.exit(main())