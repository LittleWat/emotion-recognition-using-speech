import argparse
import math
import os
import pickle
import shutil
import subprocess
from dataclasses import dataclass
from test import get_estimators_name
from typing import Dict, List

import cv2
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment, silence
from tqdm import tqdm

from emotion_recognition import EmotionRecognizer
from utils import extract_features_from_array, get_best_estimators


@dataclass
class EmotionItem:
    """Class for keeping track of an item in inventory."""
    start_mill_sec: int
    end_mill_sec: int
    result: dict

    @property
    def plot(self):
        return plt.bar(*zip(*self.result.items()), color="b")

def main():
    args = parse_args()

    myaudio = AudioSegment.from_wav(args.input_filename)
    silences = silence.detect_silence(myaudio, min_silence_len=300, silence_thresh=-30)
    core_filename =  os.path.splitext(os.path.basename(args.input_filename))[0]
    outdir = os.path.join(args.output_dir, core_filename)
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    
    sample_rate = myaudio.frame_rate
    next_start = 0


    estimators = get_best_estimators(True, emotions=args.emotions.split(","))
    estimators_str, estimator_dict = get_estimators_name(estimators)
    print(f"estimator_dict: {estimator_dict}")
    features = ["mfcc", "chroma", "mel"]
    print(f"features: {features}")
    detector = EmotionRecognizer(estimator_dict[args.model], emotions=args.emotions.split(","), features=features, verbose=1)

    emotion_items = []
    for i, a_silence in enumerate(silences):
        split_sound = myaudio[next_start:a_silence[0]]
        output_filename = os.path.join(outdir, f"{core_filename}_{i:03}.wav")
        split_sound.export(output_filename, format="wav")

        pred = detector.predict_proba_by_audio_segment(split_sound)
        emotion_items.append(EmotionItem(next_start, a_silence[0], pred))

        next_start = a_silence[1]
        
    output_mp4name = os.path.join(args.output_dir, f"{core_filename}_with_graph.mp4")
    generate_movie(emotion_items, len(myaudio), args.input_filename, output_mp4name)

def parse_args():
    parser = argparse.ArgumentParser(description="""
                                    Testing emotion recognition system using your voice, 
                                    please consider changing the model and/or parameters as you wish.
                                    """)
    parser.add_argument("input_filename") 
    parser.add_argument("-o", "--output_dir", default="split_wavs")
    parser.add_argument("-e", "--emotions", help=
                                            """Emotions to recognize separated by a comma ',', available emotions are
                                            "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (pleasant surprise)
                                            and "boredom", default is "sad,neutral,happy"
                                            """, default="angry,disgust,sad,neutral,ps,happy")
    parser.add_argument("-m", "--model", help=
                                        """
                                        The model to use, 8 models available are
                                        default is "GradientBoostingClassifier"
                                        """, default="GradientBoostingClassifier")


    # Parse the arguments passed
    args = parser.parse_args()
    print(f"args: {args}")
    return args

def generate_movie(emotion_items: List[EmotionItem], total_millisec: int, input_wavname:str, output_mp4name: str, fps=10):
    now_millisec = 0
    span = 1000 / fps

    frames = [] # for storing the generated images
    emotion_item_idx = 0
    fig = plt.figure()

    empty = {"angry":0, "disgust":0,"happy":0, "neutral":0, "ps":0, "sad":0}

    print(f"total: {total_millisec}")
    while now_millisec <= total_millisec:  
        if emotion_item_idx < len(emotion_items):
            emotion_item = emotion_items[emotion_item_idx]
            if now_millisec < emotion_item.start_mill_sec:
                frames.append(plt.bar(*zip(*empty.items())))
                now_millisec += span
            elif emotion_item.start_mill_sec <= now_millisec <= emotion_item.end_mill_sec:
                frames.append(emotion_item.plot)
                now_millisec += span
            else:
                emotion_item_idx += 1
        else:
            frames.append(plt.bar(*zip(*empty.items())))
            now_millisec += span
        print(f"{now_millisec} / {total_millisec} = {now_millisec/total_millisec}")


    print(f"len(frames): {len(frames)}")

    with open(output_mp4name.replace(".mp4",".pickle"), "wb") as f:
        pickle.dump(frames, f)

    ani = animation.ArtistAnimation(fig, frames, interval=span,blit=True)
    tmp_output_mp4name = output_mp4name + ".tmp.mp4"
    ani.save(tmp_output_mp4name)

    print(f"{tmp_output_mp4name} was generated!!!") 
    subprocess.call(f"ffmpeg -y -i {tmp_output_mp4name} -i {input_wavname} -c:v copy -c:a aac {output_mp4name}", shell=True)
    print(f"{output_mp4name} was generated!!!")

    os.remove(tmp_output_mp4name)
    print(f"{tmp_output_mp4name} was removed!!!") 


if __name__ == "__main__":   
    main()
