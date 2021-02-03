import os
import tensorflow as tf
import numpy as np
from multiprocessing import Pool
from utils import cqt_dual, _float_feature
from models import onset_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=str, default=16)
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--audio_dir', type=str, required=True)
parser.add_argument('--tfrd_dir', type=str, required=True)
parser.add_argument('--save_dir', type=float, default=0.05)

args = parser.parse_args()


def generate_test_tfrd(wav_path):
    spec = cqt_dual(wav_path)
    offset = 5
    num_data = spec.shape[1]
    spec = np.pad(spec, ((0, 0), (offset, offset), (0, 0)), 'constant')

    to_example = lambda spec: tf.train.Example(features=tf.train.Features(
        feature={'spec': _float_feature(spec)}))
    suffix = '.' + os.path.basename(wav_path).split('.')[1]

    with tf.python_io.TFRecordWriter(
            os.path.join(
                args.tfrd_dir,
                os.path.basename(wav_path).replace(suffix,
                                                   '.tfrecords'))) as w:
        for i in range(offset, offset + num_data):
            example = to_example(spec[:, i - offset:i + offset +
                                      1]).SerializeToString()
            w.write(example)


def wav_to_tfrd(wav_paths):
    p = Pool(16)
    result = p.map_async(generate_test_tfrd, wav_paths)
    result.get()
    p.close()
    p.join()


def input_fn(data_path):
    def parser(serialized_example):

        H, W, C = 356, 11, 2
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'spec':
                                               tf.FixedLenFeature([H * W * C],
                                                                  tf.float32),
                                           })
        spec = tf.cast(tf.reshape(features['spec'], [H, W, C]), tf.float32)
        return spec

    with tf.variable_scope('input_pipe'):
        dataset = tf.data.TFRecordDataset(data_path, num_parallel_reads=None)
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(map_func=parser,
                                          batch_size=args.batch_size,
                                          num_parallel_calls=None))
        dataset = dataset.prefetch(6)

    return dataset


def model_fn(features, labels, mode):

    with tf.variable_scope('onset'):
        onset_logits = onset_model(features, mode)

    old_onset_scope = ['conv_block/', 'fc/', 'lstm/', 'dense/']
    new_onset_scope = ['onset/' + x for x in old_onset_scope]
    onset_scope = {
        old_scope: new_scope
        for old_scope, new_scope in zip(old_onset_scope, new_onset_scope)
    }

    predictions = {
        'onset_probs': tf.nn.sigmoid(onset_logits, name='sigmoid_onset')
    }

    tf.train.init_from_checkpoint(args.ckpt_path, onset_scope)

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def main():

    audio_files = os.listdir(args.audio_dir)
    suffix = '.' + audio_files[0].split('.')[1]
    audio_files = [
        filename for filename in audio_files if filename.replace(
            suffix, '.onset_raw') not in os.listdir(args.save_dir)
    ]
    tfrd_files = [
        filename.replace(suffix, '.tfrecords') for filename in audio_files
    ]

    audio_files_needed = [
        filename for filename in audio_files if filename.replace(
            suffix, '.tfrecords') not in os.listdir(args.tfrd_dir)
    ]
    wav_files = [
        os.path.join(args.audio_dir, filename)
        for filename in audio_files_needed
    ]
    if wav_files:
        wav_to_tfrd(wav_files)

    sess_config = tf.ConfigProto(log_device_placement=False,
                                 allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    run_config = tf.estimator.RunConfig(session_config=sess_config,
                                        save_summary_steps=100,
                                        log_step_count_steps=500)

    model = tf.estimator.Estimator(model_fn=model_fn,
                                   config=run_config,
                                   model_dir=None)

    for tfrd_file in tfrd_files:
        print(tfrd_file)
        tfrd_path = os.path.join(args.tfrd_dir, tfrd_file)
        predictions = model.predict(input_fn=lambda: input_fn(tfrd_path))

        for idx, prediction in enumerate(predictions):
            onset_probs = prediction['onset_probs']  # [B, 7]
            if idx == 0:
                onset_probs_array = onset_probs
            else:
                onset_probs_array[-6:] += onset_probs[:-1]
                onset_probs_array = np.concatenate(
                    [onset_probs_array, onset_probs[-1:]], axis=0)

        onset_probs_array = onset_probs_array[3:-3]
        onset_probs_array[3:-3] = onset_probs_array[3:-3] / 7

        for i in range(1, 4):
            onset_probs_array[i - 1] = onset_probs_array[i - 1] / (i + 3)
            onset_probs_array[-i] = onset_probs_array[-i] / (i + 3)

        probs = onset_probs_array
        raw_path = os.path.join(args.save_dir,
                                tfrd_file.replace('.tfrecords', '.onset_raw'))
        np.savetxt(raw_path, probs, fmt='%.6f', delimiter='\t')


if __name__ == '__main__':
    main()
