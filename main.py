import argparse
import os
import time
from multiprocessing import Pool
import json
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='prepare material')
    parser.add_argument('--num_pool', default=8, type=int)
    parser.add_argument('--CUDA_ids', default='1 2', type=str)
    parser.add_argument("--input_dir", default='/home/liyan/FAIR1M/test/images/', help="input path", type=str)
    parser.add_argument("--output_dir", default='output_path/orcnn_r50_fpn_1x_ISPRS_3s_le90/',
                        help="output path", type=str)
    parser.add_argument('--config', default='work_dirs/orcnn_r50_fpn_1x_ISPRS_3s_le90/orcnn_r50_fpn_1x_ISPRS_3s_le90.py',
                        help='config file', type=str)
    parser.add_argument('--checkpoint', default='work_dirs/orcnn_r50_fpn_1x_ISPRS_3s_le90/epoch_36.pth',
                        help='checkpoint file', type=str)
    args, _ = parser.parse_known_args()
    return args


def main_worker(CUDA_id, index, num_pool, input_dir, output_dir, config, checkpoint):
    print('pool num {} CUDA id {}'.format(index, CUDA_id))
    order = 'CUDA_VISIBLE_DEVICES={} python sub_main.py --pid_index {} --num_pool {} --input_dir {} --output_dir {} ' \
            '--config {} --checkpoint {}'.format(CUDA_id, index, num_pool, input_dir, output_dir, config, checkpoint)
    print(order)
    os.system(order)
    # import  pdb
    # pdb.set_trace()


if __name__ == '__main__':
    tic = time.time()

    args = parse_args()
    # spawn mp
    pool = Pool(args.num_pool)
    CUDA_ids = args.CUDA_ids.split(' ')
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_pool = args.num_pool
    config = args.config
    checkpoint = args.checkpoint
    for i in range(len(CUDA_ids)):
        CUDA_ids[i] = int(CUDA_ids[i])
    if isinstance(CUDA_ids, int):
        num_per_CUDA = num_pool
    else:
        num_per_CUDA = int(num_pool / len(CUDA_ids))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Parent process {os.getpid()}')
    count_num = 0
    for CUDA_id in CUDA_ids:
        for i in range(num_per_CUDA):
            pool.apply_async(
                main_worker, args=(CUDA_id, i + count_num*num_per_CUDA, num_pool, input_dir, output_dir,
                                   config, checkpoint))
            time.sleep(0.1)
        count_num += 1
    # for pid in range(num_pool):
    #     pool.apply_async(main_worker, args=(pid, num_pool, input_dir, output_dir))
    #     time.sleep(0.1)
    print('Waiting for all subprocesses done...')
    pool.close()
    pool.join()
    print('All subprocesses done!')

    elapsed = time.time() - tic
    print(f'cost: {elapsed} s.')

    output_dicts = []
    for i in range(num_pool):
        with open('{}results_{}.json'.format(output_dir, i)) as f:
            output_dict = json.load(f)
            output_dicts.extend(output_dict)
    output_path = '{}results_final.json'.format(output_dir)
    # print(output_dicts)

    mmcv.dump(output_dicts, output_path, indent=True)


