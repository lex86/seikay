import os
import matplotlib.pyplot as plt
import numpy as np
import lvm_read
import librosa
import argparse
import glob
import json
from functools import partial
import multiprocessing as mp


def task(lvm_file, params):
    # read params
    channel = params['channel']
    n_fft = params['n_fft']
    win_length = params['win_length']
    hop_length = params['hop_length']
    win_type = params.get('win_type', 'hann')
    threshold = params['threshold']
    min_duration = params['min_duration'] / hop_length  # convert to frames

    print('Processing {}'.format(lvm_file))

    # read lvm file
    lvm = lvm_read.read(lvm_file)
    signal = lvm[0]['data'][:, channel]

    # normalize signal
    signal -= np.mean(signal)
    signal /= np.std(signal)

    # process signal
    fft = librosa.stft(signal,
                       n_fft=n_fft,
                       win_length=win_length,
                       hop_length=hop_length,
                       window=win_type)
    feats = np.abs(fft)
    energy = np.mean(feats, axis=0)

    # compute train mask for frames
    mask = np.array(energy > threshold, dtype=np.int)
    diff = mask[0:-1] - mask[1:]  # make difference
    changes = np.where(diff != 0)[0]  # search for changes
    changes = np.append(changes, len(mask) - 1)  # add the last sample

    # convert not train frames to train frames if sequence not train frames length < min duration
    start_index = 0
    for index in changes:
        end_index = index + 1
        if mask[start_index] == 0:
            duration = end_index - start_index
            if duration < min_duration:
                mask[start_index:end_index] = 1
        start_index = end_index

    # compute train mask for signal
    signal_mask = np.zeros_like(signal, dtype=np.int)
    diff = mask[0:-1] - mask[1:]  # make difference
    changes = np.where(diff != 0)[0]  # search for changes
    changes = np.append(changes, len(mask) - 1)  # add the last sample
    start_index = 0
    for index in changes:
        end_index = index + 1
        if mask[start_index] == 1:
            start, end = librosa.frames_to_samples([start_index, end_index],
                                                   hop_length=hop_length,
                                                   n_fft=n_fft)
            signal_mask[start:end] = 1
        start_index = end_index

    results = {lvm_file: {'signal': signal,
                          'signal_mask': signal_mask,
                          'feats': feats,
                          'energy': energy,
                          'mask': mask}}
    return results


def main(args):
    # read config file
    config = json.load(open(args.config, 'r'))

    # find lvm files
    lvm_files = []
    if os.path.isfile(args.input):
        lvm_files = [args.input]
    elif os.path.isdir(args.input):
        lvm_files = glob.glob(os.path.join(args.input, '**/*.lvm'), recursive=True)
    else:
        Exception('Wrong input type {}'.format(args.input))

    if len(lvm_files) == 0:
        Exception('No lvm files here {}'.format(args.input))
    print('Number of lvm files: {}'.format(len(lvm_files)))

    # process lvm file in parallel
    n_proc = config['n_proc']
    pool = mp.Pool(processes=n_proc)
    results = pool.map(partial(task, params=config), lvm_files)
    dict_results = {}
    for r in results:
        dict_results.update(r)

    if args.plot:
        for k, v in dict_results.items():
            fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex='col')
            fig.suptitle(k)
            ax[0].set_title('signal')
            ax[0].set_xlabel('samples')
            ax[0].plot(v['signal'])
            ax[1].set_title('signal mask')
            ax[1].set_xlabel('samples')
            ax[1].plot(v['signal_mask'])
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train detector')
    parser.add_argument('-c', dest='config', default=u'config.json', help='path to config file')
    parser.add_argument('-i', dest='input', default=u'input.lvm', help='path to lvm file or directory')
    parser.add_argument('-o', dest='output', default=u'output', help='output directory')
    parser.add_argument('-j', dest='n_proc', default=1, type=int, help='number of processes')
    parser.add_argument('-l', dest='save_lab', default=False, action='store_true', help='create lab file')
    parser.add_argument('-ly', dest='save_lab_force', default=False, action='store_true',
                        help='overwrite existing lab file')
    parser.add_argument('--lab-ext', dest='lab_ext', default=u'lab', help='lab extension')
    parser.add_argument('--plot', dest='plot', default=False, action='store_true', help='plot results')

    arguments = parser.parse_args()
    main(arguments)
