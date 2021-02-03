import argparse
import os
import mir_eval
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--label_path', type=str, required=True)
parser.add_argument('--result_path', type=str, required=True)
parser.add_argument('--onset_tolerance', type=float, default=0.05)


def eval_result(args):
    file_path = args.file_path
    if file_path and not os.path.exists(file_path):
        os.makedirs(file_path)

    label_path = args.label_path
    result_path = args.result_path
    onset_tolerance = args.onset_tolerance
    eval_files = [filename for filename in os.listdir(result_path)]
    filenames = []
    eval_result = []

    for filename in eval_files:
        ref_file = os.path.join(label_path, filename.replace('.res', '.txt'))
        est_file = os.path.join(result_path, filename)
        ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals(
            ref_file)
        est_intervals, est_pitches = mir_eval.io.load_valued_intervals(
            est_file)

        precision, recall, f1, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            mir_eval.util.midi_to_hz(ref_pitches),
            est_intervals,
            mir_eval.util.midi_to_hz(est_pitches),
            offset_ratio=None,
            onset_tolerance=onset_tolerance)

        filenames.append(filename)
        eval_result.append([f1, precision, recall])
    mean_pitch = np.mean(eval_result, axis=0)

    for filename, (f1, precision, recall) in zip(filenames, eval_result):
        print('{}\n'
              'f1: {:.4f} precision: {:.4f} recall: {:.4f}\n'.format(
                  filename, f1, precision, recall))
    print('mean_result\n'
          'f1: {:.4f} precision: {:.4f} recall: {:.4f}\n'.format(
              mean_pitch[0], mean_pitch[1], mean_pitch[2]))


if __name__ == '__main__':
    args = parser.parse_args()
    eval_result(args)
