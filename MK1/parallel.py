import sys, math
import transformers, datasets, torch, functools, pickle, multiprocessing, psutil, concurrent.futures
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from queue import Queue

def in_parallel(data, model):
    print(data)

def main():
    parser = ArgumentParser(description='Baseline: run the pretrained model on the test data to generate summaries')
    parser.add_argument('--model', '-m', help='Select the model whose vocabulary would be used.', type=str, required=True)
    parser.add_argument('--dataset', '-d', help='Pickle for the tokenized dataset.', type=Path, required=True)
    parser.add_argument('output_file', type=Path)
    args = parser.parse_args()

    output_file = args.output_file.resolve()

    # Running the model in parallel:
    # Each generate() requires full model memory allocation per example for some reason.
    # This means an input of size (N, input_size) will allocate N copies? Making it really easy to run out of memory.
    # If the model is sufficiently small, we run no more than CPU count in parallel.

    # So we limit the maximum parallel executions based on amount of memory and GPU memory.
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model)
    # Add 2% model overhead, because parameter size if crude estimate. There might be more memory required, but likely it will be O(1) on the model size.
    model_memory_size = sum(p.numel() for p in model.parameters()) * model.dtype.itemsize * 1.02
    device_list = []
    cpu_count = multiprocessing.cpu_count()
    if torch.cuda.is_available():
        gpu_device = torch.device('cuda')
        _, gpu_total_mem = torch.cuda.mem_get_info(gpu_device)
        gpu_alloc_count = math.floor(gpu_total_mem / model_memory_size)
        if gpu_alloc_count > 0:
            # GPU can execute more items in parallel, but the data must eventually be collected on the CPU.
            # So it is not productive to run more than CPU count in parallel, maybe?
            device_list.append((gpu_device, model.to(gpu_device), min(gpu_alloc_count, cpu_count)))
    total_memory = psutil.virtual_memory().total
    cpu_alloc_count = math.floor(total_memory / model_memory_size)
    if cpu_alloc_count > 0: # should always be true? maybe? If not, from_pretrained() should had exception by now
        cpu_device = torch.device('cpu')
        device_list.append((cpu_device, model.to(cpu_device), min(cpu_alloc_count, cpu_count)))
    del model
    
    if len(device_list) <= 0:
        raise RuntimeError('Not enough memory to execute a single model.generate')
    
    print([(x[0], x[2]) for x in device_list])
    return 0
    
    device_queue = Queue(maxsize=sum(info[2] for info in device_list))

    for device, model, slots in device_list:
        for _ in range(slots):
            device_queue.put((device, model))

    with open(args.dataset.resolve(), 'rb') as file:
        dataset_map = pickle.load(file)

    jobs = []
    for dataset_name in dataset_map:
        jobs += [{
            'dataset': dataset_name,
            'text': example[dataset_map[dataset_name]['attributes']['text']],
            'human_summary': example[dataset_map[dataset_name]['attributes']['summary']],
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'labels': example['labels'],
        } for example in dataset_map[dataset_name]['tokens']['test']]

    # def process_example(job):
    #     device, model = device_queue.get()
    #     try:
    #         return model.generate(
    #             input_ids=torch.LongTensor(job['input_ids']).to(device)[None],
    #             attention_mask=torch.LongTensor(job['attention_mask']).to(device)[None],
    #             max_length=180,
    #             min_length=10,
    #             num_beams=4,
    #             length_penalty=2.0,
    #             use_cache=True
    #         )
    #     finally:
    #         device_queue.put(device, model)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=device_queue.maxsize) as pool:
    #     futures = [pool.submit(process_example, job) for job in jobs]
    #     tqdm(concurrent.futures.as_completed(futures), desc='Genearating summaries', total=len(futures))

    
    # concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    # result_key = f'model-{args.model}-output_ids'
    # for index, future in enumerate(futures):
    #     jobs[index][result_key] = future.result()

    return 0

if __name__ == '__main__':
    sys.exit(main())