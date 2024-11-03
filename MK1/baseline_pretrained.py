import sys
import transformers, datasets, torch, functools, pickle, multiprocessing
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

def main():
    with open('dataset-t5-base.pickle', 'rb') as file:
        dataset_map = pickle.load(file)

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
    output_device = torch.device('cpu')
    model_run_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = transformers.T5ForConditionalGeneration.from_pretrained('t5-base').to(model_run_device)

    for dataset_name in dataset_map:
        input_ids = dataset_map[dataset_name]['test']['input_ids'][:100]
        input_attention_mask = dataset_map[dataset_name]['test']['input_attention_mask'][:100]
        output = []
        for index in tqdm(range(0, 100), unit='examples', desc=f'Generating summaries [{dataset_name}]'):
            input_ids = tokenizer.encode(dataset_map[dataset_name]['test']['text'][index], return_tensors='pt').to(model_run_device)
            summary_output_ids = model.generate(
                input_ids=input_ids,
                max_length=192,
                min_length=10,
                num_beams=4,
                length_penalty=2.0,
                use_cache=False
            ).to(output_device)
            summary_text = tokenizer.decode(summary_output_ids[0], skip_special_tokens=True)
            output.append({
                'text': dataset_map[dataset_name]['test']['text'][index],
                'summary': dataset_map[dataset_name]['test']['summary'][index],
                'gen_summary': summary_text,
                'input_text': tokenizer.decode(input_ids[0], skip_special_tokens=True),
                'input_ids': input_ids[0],
                'output_ids': summary_output_ids[0]
            })
        with open(f'pretrained-{dataset_name}-t5-base.pickle', 'wb') as file:
            pickle.dump(output, file)

    return 0

if __name__ == '__main__':
    sys.exit(main())