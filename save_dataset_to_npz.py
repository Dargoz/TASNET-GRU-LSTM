import subprocess
from scipy.io import wavfile
#import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import signal
#import stft
#import h5py
import librosa
import argparse
import yaml


SR = 8000
counter = 0
    
def processAudio(subdir, speaker):
    global counter
    audioPath = os.path.join(CLEAN_INPUT_DIR, subdir, speaker)
    for audio in os.listdir(audioPath):
        if audio[-4:] != '.WAV':
            continue
        # print(audio)
        filePath = os.path.join(CLEAN_INPUT_DIR, subdir, speaker, audio)
        print('process '+filePath)

        data, sr = librosa.load(filePath, sr=SR) # Downsample to 8kHz
        
        processed_data.append([speaker, audio, data])

        counter += 1
        # if counter == 5:
        #     return -1

def processSpeaker(subdir):
    speakerPath = os.path.join(CLEAN_INPUT_DIR, subdir)
    for speaker in os.listdir(speakerPath):
        print(speaker)
        if processAudio(subdir, speaker) == -1:
            return -1

def parse_yaml(yaml_conf):
    if not os.path.exists(yaml_conf):
        raise FileNotFoundError(
            "Could not find config file...{}".format(yaml_conf))
    with open(yaml_conf, 'r') as f:
        config_dict = yaml.load(f)
    return config_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Data generation by PyTorch ")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        dest="config",
        help="Location of .yaml configure files for training")
    args = parser.parse_args()
    config_dict = parse_yaml(args.config)
    data_config = config_dict["data_generator"]

    train_data_dir = os.path.join(data_config['clean_speech_dir'],'train','files')
    train_npz_dir = os.path.join(data_config['clean_speech_dir'], 'train','npz')
    validation_data_dir = os.path.join(data_config['clean_speech_dir'],'validation','files')
    validation_npz_dir = os.path.join(data_config['clean_speech_dir'],'validation','npz')
    SR = data_config['sr']

    processed_data = []
    
    print('#########Convert train data#########')
    for dir in os.listdir(train_data_dir):
        print(dir)
        if processSpeaker(dir) == -1:
            break

    print('{} data is processed'.format(counter))
    npzFilename = train_npz_dir+'/clean.npz'
    np.savez(npzFilename, data=processed_data, sr=SR)
    print(npzFilename+' is created')


    print('#########Convert validation data#########')
    global counter = 0
    for dir in os.listdir(validation_data_dir):
        print(dir)
        if processSpeaker(dir) == -1:
            break

    print('{} data is processed'.format(counter))
    npzFilename = validation_npz_dir+'/clean.npz'
    np.savez(npzFilename, data=processed_data, sr=SR)
    print(npzFilename+' is created')

    # npzfile = np.load(NPZ_OUTPUT_DIR+'/clean.npz', allow_pickle=True)
    # print(npzfile.files)
    # print(npzfile['data'])
    # print(npzfile['sr'])