import subprocess
from scipy.io import wavfile
import numpy as np
from scipy import signal
#import stft
#import h5py
from pathlib import Path
import os
import argparse
import yaml
import librosa
import math
import random

def parse_yaml(yaml_conf):
    if not os.path.exists(yaml_conf):
        raise FileNotFoundError(
            "Could not find config file...{}".format(yaml_conf))
    with open(yaml_conf, 'r') as f:
        config_dict = yaml.load(f)
    return config_dict
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Setup")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        dest="config",
        help="Location of .yaml configure files for training")
    args = parser.parse_args()
    config_dict = parse_yaml(args.config)
    data_config = config_dict["data_generator"]

    datasetBaseDir = data_config['dataset_base_dir']
    # timitDatasetDir = datasetBaseDir+data_config['raw_timit_dir']+'/TRAIN'
    timitDatasetDir = datasetBaseDir+data_config['raw_timit_dir']+'/TEST'
    # outputDataDir = os.path.join(data_config['dataset_base_dir']+data_config['clean_speech_dir'],'train')
    outputDataDir = os.path.join(data_config['dataset_base_dir']+data_config['clean_speech_dir'],'validation')
    filesOutputDir = outputDataDir+'/files'
    npzOutputDir = outputDataDir+'/npz'

    print(datasetBaseDir)
    print(timitDatasetDir)
    print(filesOutputDir)
    print(npzOutputDir)

    #param
    ##train
    # totalTimitSpeaker = 462
    # audioPerSpeakerLimit = 1
    # totalSpeakerLimit = 270

    ##test
    totalTimitSpeaker = 168
    audioPerSpeakerLimit = 1
    totalSpeakerLimit = 168


    processed_data = []
    speakerCount = 0
    totalSpeakerCount = 0
    totalWomanSpeaker = 0
    totalManSpeaker = 0
    audioCount = 0
    tempRemainingAdditionalSpeaker = 0

    timitDataset = os.listdir(timitDatasetDir)
    percentageSpeakerPerDialect = totalSpeakerLimit / totalTimitSpeaker

    print('percentage speakers per dialect : {}'.format(percentageSpeakerPerDialect))

    for dir in timitDataset:
        print()
        print(dir) 
        speakerSet = os.listdir(os.path.join(timitDatasetDir, dir))
        womanSpeakerSet = [audio for audio in speakerSet if audio[0] == 'F']
        manSpeakerSet = [audio for audio in speakerSet if audio[0] == 'M']

        print('{} woman speakers available for dialect {}'.format(len(womanSpeakerSet),dir))
        print('{} man speakers available for dialect {}'.format(len(manSpeakerSet),dir))

        speakerLimitPerDialect = math.ceil(percentageSpeakerPerDialect * len(speakerSet))
        print('total {} speakers taken for dialect {}'.format(speakerLimitPerDialect,dir))

        womanSpeakerLimitPerDialect = math.ceil(speakerLimitPerDialect/2)
        if len(womanSpeakerSet) < womanSpeakerLimitPerDialect:
            manSpeakerLimitPerDialect = math.ceil(speakerLimitPerDialect/2) + womanSpeakerLimitPerDialect - len(womanSpeakerSet)
            womanSpeakerLimitPerDialect = len(womanSpeakerSet)
        else:
            manSpeakerLimitPerDialect = math.ceil(speakerLimitPerDialect/2)

        ###
        if tempRemainingAdditionalSpeaker > 0:
            manSpeakerLimitPerDialect += tempRemainingAdditionalSpeaker
            tempRemainingAdditionalSpeaker = 0

        if manSpeakerLimitPerDialect > len(manSpeakerSet):
            tempRemainingAdditionalSpeaker += manSpeakerLimitPerDialect-len(manSpeakerSet)
            manSpeakerLimitPerDialect = len(manSpeakerSet)
        ###


        totalWomanSpeaker += womanSpeakerLimitPerDialect
        totalManSpeaker += manSpeakerLimitPerDialect
        print('{} woman speakers taken for dialect {}'.format(womanSpeakerLimitPerDialect,dir))
        print('{} man speakers taken for dialect {}'.format(manSpeakerLimitPerDialect,dir))

        womanSpeakerSet = random.sample(womanSpeakerSet, k=womanSpeakerLimitPerDialect)
        manSpeakerSet = random.sample(manSpeakerSet, k=manSpeakerLimitPerDialect)
        speakerSet = womanSpeakerSet
        print('{} woman speakers for dialect {}'.format(womanSpeakerSet,dir))

        speakerSet.extend(manSpeakerSet)
        print('{} man speakers for dialect {}'.format(manSpeakerSet,dir))
        
        for speaker in speakerSet:
            # print(speaker)

            audioSet = os.listdir(os.path.join(timitDatasetDir, dir, speaker))
            audioSet = [audioFile for audioFile in audioSet if audioFile[-4:] == '.WAV']
            audioSet = random.sample(audioSet, k=audioPerSpeakerLimit)
            print(audioSet)

            for audio in audioSet:
                filePath = os.path.join(timitDatasetDir, dir, speaker,audio)
                # print(filePath)

                outputFolder = '{}/{}/{}'.format(filesOutputDir, dir, speaker)  
                Path(outputFolder).mkdir(parents=True, exist_ok=True)
                
                subprocess.run(["sph2pipe", "-f", "wav", filePath, outputFolder+'/'+audio])

                data, sr = librosa.load(filePath, sr=data_config['sr']) # Downsample to 8kHz
                
                processed_data.append([speaker, audio, data])

                audioCount += 1
                # if audioCounter%audioPerSpeakerLimit==0:
                #     break

            speakerCount+=1
            speakerTest = os.listdir('{}/{}/{}'.format(filesOutputDir, dir, speaker))
            if len(speakerTest) > audioPerSpeakerLimit:
                raise ValueError('{} has more than {} audio'.format(speaker,audioPerSpeakerLimit))
            # totalSpeakerCount+=1
            # if speakerCount == speakerLimitPerDialect:
            #     speakerCount=0
            #     break

    np.savez(npzOutputDir+'/clean.npz', data=processed_data, sr=data_config['sr'])
    print('total speakers'.format(speakerCount))
    print('{} man'.format(totalManSpeaker))
    print('{} woman'.format(totalWomanSpeaker))
    print('man : woman = {} : {}'.format(totalManSpeaker/(totalManSpeaker+totalWomanSpeaker), totalWomanSpeaker/(totalManSpeaker+totalWomanSpeaker)))
    print('{} data is processed'.format(len(processed_data)))