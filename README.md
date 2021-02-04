# Transition-aware
**Transition-aware** is an RCNN-based piano transcription model, inferring onset, pitch of 88 notes from raw audios. Compared with the existing models, like [**Onsets and Frames**](https://github.com/magenta/magenta/tree/master/magenta/models/onsets_frames_transcription) and [**High-resolution**](https://github.com/qiuqiangkong/piano_transcription_inference), Transition-aware is a more stable and robust approach.

In this repository, the **inference code** of Transition-aware model and a sample of **OMAPS dataset** are provided. The complete code and dataset will be available when the paper is published.

## Installation
Transition-aware model is developed with python3.6, tensorflow-gpu1.10, and other python packages. The development environment dependencies can be easily installed simply by running the following command:
```shell
pip install -r requirements.txt
```

## Usage
1. Transcribe the audio activation status probabilities.
  ```shell
  CKPT_PATH=<Directory containing the model checkpoint files>
  AUDIO_DIR=<Directory containing the raw audio files>
  TFRD_DIR=<Directory saving the temporal tfrecord files>
  SAVE_DIR=<Directory saving the audio activation probs>
  python inference/predict_onset.py --ckpt_path="${CKPT_PATH}" --audio_dir="${AUDIO_DIR}" --tfrd_dir="${TFRD_DIR}" --save_dir="${SAVE_DIR}"
  ```
2. Transcribe the onset probabilities of 88 notes.
  ```shell
  CKPT_PATH=<Directory containing the model checkpoint files>
  AUDIO_DIR=<Directory containing the raw audio files>
  TFRD_DIR=<Directory saving the temporal tfrecord files>
  SAVE_DIR=<Directory saving the onset probs>
  python inference/predict_pitch.py --ckpt_path="${CKPT_PATH}" --audio_dir="${AUDIO_DIR}" --tfrd_dir="${TFRD_DIR}" --save_dir="${SAVE_DIR}"
  ```
3. Using peak selection to get the final (onset, pitch) sequence from the audio status probs and onset probs.
  ```shell
  RESULT_PATH=<Directory saving results>
  PITCH_PATH=<Directory containing onset probs>
  ONSET_PITCH=<Directory cotianing audio status probs>
  python inference/probs2res.py --result_path="${RESULT_PATH}" --pitch_path="${PITCH_PATH}" --onset_path="${ONSET_PITCH}"
  ```
**The transcription results of MAPS dataset** are provided in [result/MAPS](result/MAPS).

By the way, we also provide code, [inference/eval.py](inference/eval.py)  to evaluate transcription results using mir_eval library. And [inference/midi2txt.py](inference/midi2txt.py) is provided for decoding onset, offset, pitch,  and velocity from midi files.

## OMAPS dataset
**OMAPS** dataset stands for **Ordinary MIDI Aligned Piano Sounds** dataset. OMAPS dataset is recorded from an ordinary electronic Yamaha piano P-115 played by a professional musician under the general recording environment. The MIDI derived from the electronic piano is used as the annotation. This dataset contains 106 pieces of easy, medium and difficult scores at different levels, a total of 216 minutes. It is used to evaluate the performance of piano transcription algorithms in an **ordinary recording environment**.

The complete OMAPS dataset will be available when the paper is published. A sample of this dataset is provided in directory [inference/omaps](inference/omaps). The three columns of label file in inference/omaps/sample.txt represent onset, offset, and pitch respectively.

<audio id="audio" controls="" preload="none">
   <source id="mp3" src="omaps/sample.mp3">
</audio>