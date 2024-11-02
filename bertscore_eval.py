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
        predictions['samsum'] = pickle.load(file)
    
    bertscore = evaluate.load('bertscore')

    fig, axes = plt.subplots(ncols=8, figsize=(8, 1))
    i = 0

    medians = []

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
            boxplot = axes[i].boxplot(
                [score['precision'], score['recall']],
                tick_labels=['precision', 'recall']
            )
            medians.append([numpy.mean(line.get_ydata()) for line in boxplot['medians']])
            if trained_dataset == 'pretrained':
                trained_label = 'Pretrained Model'
            else:
                trained_label = f'Full fine-tuned\n on {trained_dataset.upper()}'
            axes[i].set_title(f'{trained_label},\nsummarizing {test_dataset.upper()}', fontsize=8, rotation=90, ha='left')
            axes[i].set_ylabel('BERT Score')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=90)
            for key in ['precision', 'recall']:
                # print(f'{trained_dataset}.{test_dataset}.{key}.mean = {numpy.mean(score[key])}')
                print(f'{trained_dataset}.{test_dataset}.{key}.median = {numpy.median(score[key])}')
            i += 1
    print(medians)
    plt.subplots_adjust(wspace=1)
    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())
