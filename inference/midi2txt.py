import argparse
import pretty_midi
import os
from glob import glob
pretty_midi.pretty_midi.MAX_TICK = 1e10

parser = argparse.ArgumentParser()
parser.add_argument('--midi_path', type=str, required=True)
parser.add_argument('--txt_path', type=str, required=True)


def extract_labels_from_midi(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    outputs = []
    for instrument in midi_data.instruments:
        notes = instrument.notes
        for note in notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            outputs.append([start, end, pitch])
    outputs.sort(key=lambda elem: elem[0])
    return outputs


def convert_midi_to_txt(args):
    midi_path = args.midi_path
    txt_path = args.txt_path
    notes = extract_labels_from_midi(midi_path)
    with open(txt_path, 'wt', encoding='utf8') as f:
        for (onset, offset, pitch) in notes:
            f.write("{:.6f}\t{:.6f}\t{}\n".format(onset, offset, pitch))


if __name__ == '__main__':
    args = parser.parse_args()
    convert_midi_to_txt(args)
