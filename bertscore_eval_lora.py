import sys
import transformers, datasets, torch, evaluate, functools, pickle, multiprocessing, numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

def main():
    predictions = {}
    with open('lora-fine-tuning-predictions-cnn.pickle', 'rb') as file:
        predictions['cnn'] = pickle.load(file)

    with open('lora-fine-tuning-predictions-samsun.pickle', 'rb') as file:
        predictions['samsun'] = pickle.load(file)
    
    bertscore = evaluate.load('bertscore')


    fig, axes = plt.subplots(ncols=6, figsize=(6, 1))
    i = 0

    for test_dataset in predictions:
        for trained_dataset in predictions[test_dataset]:
            if trained_dataset == 'references':
                continue
            score = bertscore.compute(
                predictions=predictions[test_dataset][trained_dataset],
                references=predictions[test_dataset]['references'],
                lang='en',
                model_type='t5-base'
            )
            axes[i].boxplot(
                [score['precision'], score['recall']],
                tick_labels=['precision', 'recall']
            )
            if trained_dataset == 'pretrained':
                trained_label = 'Pretrained Model'
            else:
                trained_label = f'LoRA fine-tuned model on {trained_dataset.upper()}'
            axes[i].set_title(f'{trained_label},\nsummarizing {test_dataset.upper()} dataset', fontsize=8)
            axes[i].set_ylabel('BERT Score')
            axes[i].set_ylim(0, 1)
            for key in ['precision', 'recall']:
                # print(f'{trained_dataset}.{test_dataset}.{key}.mean = {numpy.mean(score[key])}')
                print(f'{trained_dataset}.{test_dataset}.{key}.median = {numpy.median(score[key])}')
            i += 1
    plt.subplots_adjust(wspace=1)
    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())
