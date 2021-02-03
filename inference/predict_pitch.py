import os
import tensorflow as tf
import numpy as np
from multiprocessing import Pool
from utils import cqt_dual, params
from utils import _bytes_feature, normalize_np
from models import normalize, acoustic_model, lstm_layer, fc_layer

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=str, default=16)
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--audio_dir', type=str, required=True)
parser.add_argument('--tfrd_dir', type=str, required=True)
parser.add_argument('--save_dir', type=float, default=0.05)

args = parser.parse_args()


def generate_test_tfrd(wav_path):
    spec = cqt_dual(wav_path)
    max_len = params.max_len

    data_len = int(np.ceil(spec.shape[1] / max_len) * max_len)
    end_pad = data_len - spec.shape[1]
    data_num = int(data_len / max_len)
    spec = np.pad(spec, ((0, 0), (0, end_pad), (0, 0)))  #[356, len, 2]

    pad_len = 4
    spec = np.pad(spec, ((0, 0), (pad_len, pad_len), (0, 0)))

    spec = normalize_np(spec)

    to_example = lambda spec: tf.train.Example(features=tf.train.Features(
        feature={'spec': _bytes_feature(spec.tobytes())}))
    spec_offset = int((max_len + 8) / 2)

    suffix = '.' + os.path.basename(wav_path).split('.')[1]

    with tf.python_io.TFRecordWriter(
            os.path.join(
                args.tfrd_dir,
                os.path.basename(wav_path).replace(suffix,
                                                   '.tfrecords'))) as w:
        for i in range(data_num):
            j = spec_offset + i * max_len
            example = to_example(spec[:, j - spec_offset:j + spec_offset +
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

        H, W, C = 356, params.max_len + 8, 2
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'spec':
                                               tf.FixedLenFeature([],
                                                                  tf.string),
                                           })
        spec = tf.decode_raw(features['spec'], tf.int16)
        spec = tf.reshape(spec, [H, W, C])
        return spec

    with tf.variable_scope('input_pipe'):
        dataset = tf.data.TFRecordDataset(data_path, num_parallel_reads=None)
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(map_func=parser,
                                          batch_size=batch_size,
                                          num_parallel_calls=None))
        dataset = dataset.prefetch(6)

    return dataset


def model_fn(features, labels, mode):

    with tf.variable_scope('pitch'):
        inputs = normalize(features)
        with tf.variable_scope('pitch'):
            net1 = acoustic_model(inputs, mode)  # [batch_size, time_len, 1024]
            net1 = lstm_layer(net1, 512, mode)  # [batch_size, time_len, 1024]
            pitch_logits = fc_layer(net1, 88)

    old_pitch_scope = ['pitch/']

    new_pitch_scope = ['pitch/' + x for x in old_pitch_scope]
    pitch_scope = {
        old_scope: new_scope
        for old_scope, new_scope in zip(old_pitch_scope, new_pitch_scope)
    }

    predictions = {
        'pitch_probs':
        tf.nn.sigmoid(pitch_logits,
                      name='sigmoid_pitch')  # [Batch_size, len, 88]
    }

    model_paths = args.ckpt_path
    tf.train.init_from_checkpoint(model_paths, pitch_scope)

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def main():

    audio_files = os.listdir(args.audio_dir)
    suffix = '.' + audio_files[0].split('.')[1]

    audio_files = [
        filename for filename in audio_files if filename.replace(
            suffix, '.pitch_raw') not in os.listdir(args.save_dir)
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

        pitch_probs_array = []

        for idx, prediction in enumerate(predictions):

            pitch_probs = prediction['pitch_probs']
            pitch_probs_array.append(pitch_probs)

        pitch_probs_array = np.concatenate(pitch_probs_array, axis=0)

        probs = pitch_probs_array
        raw_path = os.path.join(args.save_dir,
                                tfrd_file.replace('.tfrecords', '.pitch_raw'))
        np.savetxt(raw_path, probs, fmt='%.6f', delimiter='\t')


if __name__ == '__main__':
    main()
