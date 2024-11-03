import sys
import transformers, datasets, torch, functools, pickle, multiprocessing
from argparse import ArgumentParser
from pathlib import Path

def main():
    parser = ArgumentParser(description='Tokenize the target datasets and store the result into pickle')
    parser.add_argument('--model', '-m', help='Select the model whose vocabulary would be used.', type=str, required=True)
    parser.add_argument('output_file', type=Path)
    args = parser.parse_args()

    output_file = args.output_file.resolve()

    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model)
    news_dataset = datasets.load_dataset('abisee/cnn_dailymail', '3.0.0')
    samsun_dataset = datasets.load_dataset('Samsung/samsum')
    dataset_map = {
        'samsun': {
            'data': samsun_dataset,
            'attributes': {
                'text': 'dialogue',
                'summary': 'summary'
            }
        },
        'cnn': {
            'data': news_dataset,
            'attributes': {
                'text': 'article',
                'summary': 'highlights'
            }
        }
    }

    def tokenize_dataset(examples, *, text_attribute, summary_attribute, input_max_length=512, summary_max_length=128):
        inputs = ["summarize: " + doc for doc in examples[text_attribute]]
        model_inputs = tokenizer(
            inputs,
            max_length=input_max_length,
            truncation=transformers.tokenization_utils_base.TruncationStrategy.LONGEST_FIRST,
            padding=transformers.utils.PaddingStrategy.MAX_LENGTH,
            is_split_into_words=False,
            return_tensors=transformers.utils.TensorType.PYTORCH,
            return_attention_mask=True
        )
        human_summary = tokenizer(
            examples[summary_attribute],
            max_length=summary_max_length,
            truncation=transformers.tokenization_utils_base.TruncationStrategy.LONGEST_FIRST,
            padding=transformers.utils.PaddingStrategy.MAX_LENGTH,
            is_split_into_words=False,
            return_tensors=transformers.utils.TensorType.PYTORCH,
            return_attention_mask=True
        )
        return {
            'input_ids': model_inputs['input_ids'],
            'input_attention_mask': model_inputs['attention_mask'],
            'ref_summary_ids': human_summary['input_ids'],
            'ref_summary_attention_mask': human_summary['attention_mask']
        }
    
    kwargs = {}
    cpu_count = multiprocessing.cpu_count()
    if isinstance(cpu_count, int) and cpu_count > 1:
        kwargs['num_proc'] = cpu_count

    data = {}
    for name in dataset_map:
        print('Tokenizing dataset: %s' % name)
        data[name] = {}
        tokenized = dataset_map[name]['data'].map(functools.partial(tokenize_dataset, text_attribute=dataset_map[name]['attributes']['text'], summary_attribute=dataset_map[name]['attributes']['summary']), batched=True, **kwargs)
        for usage in tokenized.keys():
            data[name][usage] = {}
            for key in ['input_ids', 'input_attention_mask', 'ref_summary_ids', 'ref_summary_attention_mask']:
                data[name][usage][key] = torch.LongTensor(tokenized[usage][key])
            for target_key, source_key in dataset_map[name]['attributes'].items():
                data[name][usage][target_key] = tokenized[usage][source_key]

    output_file: Path
    output_file.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)

    return 0

if __name__ == '__main__':
    sys.exit(main())