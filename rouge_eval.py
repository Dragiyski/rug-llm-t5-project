import sys
import transformers, datasets, torch, evaluate, functools, pickle, multiprocessing, numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

def main():
    predictions = {}
    with open('baseline-full-fine-tuning-predictions-cnn.pickle', 'rb') as file:
        predictions['cnn'] = pickle.load(file)

    with open('baseline-full-fine-tuning-predictions-samsun.pickle', 'rb') as file:
        predictions['samsun'] = pickle.load(file)
    
    rouge = evaluate.load('rouge')

    fig, axes = plt.subplots(ncols=8, figsize=(8, 1))
    i = 0

    for test_dataset in predictions:
        for trained_dataset in predictions[test_dataset]:
            if trained_dataset == 'references':
                continue
            score = rouge.compute(
                predictions=predictions[test_dataset][trained_dataset],
                references=predictions[test_dataset]['references'],
            )
            axes[i].boxplot(
                [score['rougeLsum']],
                tick_labels=['ROUGE-L-Sum']
            )
            if trained_dataset == 'pretrained':
                trained_label = 'Pretrained Model'
            else:
                trained_label = f'Full fine-tuned model on {trained_dataset.upper()}'
            axes[i].set_title(f'{trained_label},\nsummarizing {test_dataset.upper()} dataset', fontsize=8)
            axes[i].set_ylabel('ROUGE Score')
            axes[i].set_ylim(0, 1)
            print(f'{trained_dataset}.{test_dataset}.rouge = {score["rougeLsum"]}')
            i += 1
    plt.subplots_adjust(wspace=1)
    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())
