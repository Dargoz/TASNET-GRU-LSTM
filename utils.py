import librosa
import numpy as np
import fnmatch
import os
import random
import time
import sklearn.utils as sku
import scipy.signal
import numpy.random
from scipy.io import wavfile
import argparse
import yaml
# import torch
from sklearn import preprocessing

from random import randrange
from math import ceil
from math import floor
from pathlib import Path

def find_files(directory, pattern=['*.wav', '*.WAV']):
    '''find files in the directory'''
    files = []
    print("Before : ", files)
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern[0]):
            files.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, pattern[1]):
            files.append(os.path.join(root, filename))
    print("After : ", files)
    return files

def find_dir(directory, pattern):
	dir = []
	for root, dirnames, filenames in os.walk(directory):
		if root.split('/')[-2] == pattern:
			dir.append(root)
	return dir		
	#print(dir)


def parse_yaml(yaml_conf):
    if not os.path.exists(yaml_conf):
        raise FileNotFoundError(
            "Could not find config file...{}".format(yaml_conf))
    with open(yaml_conf, 'r') as f:
        config_dict = yaml.load(f)
    return config_dict

def norm_audio(audiofiles, noisefiles):
	'''Normalize the audio files
	used before training using a independent script'''
	for file in audiofiles:
		audio, sr = librosa.load(file, sr=16000)
		div_fac = 1 / np.max(np.abs(audio)) / 3.0
		audio = audio * div_fac
		librosa.output.write_wav(file, audio, sr)
	for file in noisefiles:
		audio, sr = librosa.load(file, sr=16000)
		div_fac = 1 / np.max(np.abs(audio)) / 3.0
		audio = audio * div_fac
		librosa.output.write_wav(file, audio, sr)

def mix(speech1,speech2,SNR):
    len1 = len(speech1)
    len2 = len(speech2)
    tot_len = max(len1, len2)
    
    if len1 < len2:
        rep = int(np.floor(len2 / len1))
        left = len2 - len1 * rep
        temp_audio = np.tile(speech1, [1, rep])
        temp_audio.shape = (temp_audio.shape[1],)
        speech1 = np.hstack((temp_audio, speech1[:left]))
        speech2 = np.array(speech2)
    else:
        rep = int(np.floor(len1 / len2))
        left = len1 - len2 * rep
        temp_noise = np.tile(speech2, [1, rep])
        temp_noise.shape = (temp_noise.shape[1],)
        speech2 = np.hstack((temp_noise, speech2[:left]))
        speech1 = np.array(speech1)
        
    fac = np.linalg.norm(speech1)/np.linalg.norm(speech2)/(10**(SNR*0.05))
    speech2 *= fac
    mix_speech = speech1 + speech2
    return mix_speech, speech1, speech2

# def mix_audio(samples1, samples2, sr):
# 	#Find length of longest signal
# 	maxlength = max(len(samples1),len(samples2))

# 	#Pad each signal to the length of the longest signal
# 	samples1 = np.pad(samples1, (0, maxlength - len(samples1)), 'constant', constant_values=(0))
# 	samples2 = np.pad(samples2, (0, maxlength - len(samples2)), 'constant', constant_values=(0))

# 	#combine series together
# 	mixed_series = samples1 + samples2

# 	#Pad 3 wav files to whole number of seconds
# 	extrapadding = (ceil(len(mixed_series) / sr) * sr) - len(mixed_series)
# 	mixed_series = np.pad(mixed_series, (0,extrapadding), 'constant', constant_values=(0))
# 	# samples1 = np.pad(samples1, (0,extrapadding), 'constant', constant_values=(0))
# 	# samples2 = np.pad(samples2, (0,extrapadding), 'constant', constant_values=(0))
# 	return mixed_series


def normalize_mean(x):
	return (x-x.mean())/x.std()

def unnormalize(x1,x2):
	return x1*x2.std()+x2.mean()

def zero_mean(x):
	return x - x.mean()

def l2_norm(x):
	return preprocessing.scale(x,axis=1,with_mean=False,with_std=True)

def make_same_length(speech,len_ref):
	len_ref = int(len_ref)
	len_s = len(speech)
	if len_s < len_ref:
		rep = int(np.floor(len_ref / len_s))
		left = len_ref - len_s * rep
		temp_speech = np.tile(speech, [1, rep])
		temp_speech.shape = (temp_speech.shape[1],)
		speech = np.hstack((temp_speech, speech[:left]))
	else:
		rep = int(np.floor(len_s / len_ref))
		add = len_ref * (rep+1) - len_s
		speech = np.hstack((speech, speech[:add]))
	return speech

def padlast(x, len_ref):
	len_x = len(x)
	fac = int(np.floor(len_x/len_ref))
	x_padlast = np.zeros((fac+1)*len_ref)
	x_padlast[:fac*len_ref] = x[:fac*len_ref]
	x_padlast[fac*len_ref:] = x[(len_x-len_ref):]
	return np.float32(x_padlast)

def padding(x, length):
	len_x = len(x)
	fac = int(np.floor(len_x/length))
	x_padded = np.zeros((fac+1)*length)
	x_padded[:len_x] = x
	return np.float32(x_padded)
        

class speech_preprocess(object):
	def __init__(self,
				 dataset_base_dir,
				 clean_speech_dir,
				 mixed_files_save_path,
				 raw_timit_dir,
				 train_save_path,
				#  dev_save_path,
				 valid_save_path,
				 num_data,
				 trainSetPercentage,
				 sr=8000,
				 N_L=40,
				 len_time=0.5,
				 is_norm=True,
				 ):
		self.dataset_base_dir = dataset_base_dir
		self.clean_speech_dir = clean_speech_dir
		self.mixed_files_save_path = mixed_files_save_path
		self.train_save_path = dataset_base_dir + train_save_path + '/'+str(len_time)
		# self.dev_save_path = dev_save_path
		self.valid_save_path = dataset_base_dir + valid_save_path + '/'+str(len_time)
		self.num_data = num_data
		self.sr = sr
		self.N_L = N_L
		self.len_time = len_time
		self.is_norm = is_norm
		self.trainSetPercentage = trainSetPercentage
		self.trainSet = 0
		self.validSet = 0

	"""
	def speech_segment(self,mix_dir,speech1_dir,speech2_dir):
		mix, _ = librosa.load(mix_dir, self.sr)
		speech1, _ = librosa.load(speech1_dir, self.sr)
		speech2, _ = librosa.load(speech2_dir, self.sr)

		len_mix = len(mix)
		len_ref = int(self.len_time*self.sr)

		if len_mix < len_ref:
			mix = make_same_length(mix, len_ref)
			speech1 = make_same_length(speech1,len_ref)
			speech2 = make_same_length(speech2,len_ref)
		else:
			mix = padlast(mix, len_ref)
			speech1 = padlast(speech1,len_ref)
			speech2 = padlast(speech2,len_ref)
		len_tot = len(mix)
		mix = np.reshape(mix, [int(len_tot/len_ref),int(len_ref/self.N_L),self.N_L])
		speech1 = np.reshape(speech1, [int(len_tot/len_ref),int(len_ref/self.N_L),self.N_L])
		speech2 = np.reshape(speech2, [int(len_tot/len_ref),int(len_ref/self.N_L),self.N_L])
		return mix, speech1, speech2
	"""

	def segment_audio(self,mix,speech1,speech2):
		len_mix = len(mix)
		len_ref = int(self.len_time*self.sr)

		# print(len(mix))
		# print(len(speech1))
		# print(len(speech2))
		# print("-----")

		if len_mix < len_ref:
			mix = make_same_length(mix, len_ref)
			speech1 = make_same_length(speech1,len_ref)
			speech2 = make_same_length(speech2,len_ref)
		else:
			mix = padlast(mix, len_ref)
			speech1 = padlast(speech1,len_ref)
			speech2 = padlast(speech2,len_ref)
		len_tot = len(mix)

		# print(len(mix))
		# print(len(speech1))
		# print(len(speech2))

		mix = np.reshape(mix, [int(len_tot/len_ref),int(len_ref/self.N_L),self.N_L])
		speech1 = np.reshape(speech1, [int(len_tot/len_ref),int(len_ref/self.N_L),self.N_L])
		speech2 = np.reshape(speech2, [int(len_tot/len_ref),int(len_ref/self.N_L),self.N_L])
		# print(mix.shape)
		# print(speech1.shape)
		# print(speech2.shape)
		
		return mix, speech1, speech2

	def save_speech(self,mix_speech,speech1,speech2,snr,filename):
		num_speech = mix_speech.shape[0]
		partIdx = list(range(0, num_speech))
		trainSetIdx = random.sample(partIdx, k=floor(self.trainSetPercentage*num_speech))
		self.trainSet += len(trainSetIdx)
		self.validSet += num_speech-len(trainSetIdx)
		
		print('save {} (train:{}/ valid:{}) segmented data'.format(num_speech, len(trainSetIdx), num_speech-len(trainSetIdx)))
		for ind in range(num_speech):
			if ind in trainSetIdx:
				filePath = self.train_save_path
			else:
				filePath = self.valid_save_path
			finalFilePath = '{}/{}_segment-{}.npz'.format(filePath, filename, str(ind))
			np.savez(finalFilePath,
			 		 mix_speech=mix_speech[ind,:,:],
			 		 speech1=speech1[ind,:,:],
			 		 speech2=speech2[ind,:,:],
					 snr=snr)
			
			#test load
			testData = np.load(finalFilePath, allow_pickle=True)
			testData.close()

	#TODO
	def save_test_speech(self,mix_speech,speech1,speech2,snr,save_path):
		num_speech = mix_speech.shape[0]

		# print(num_speech)
		print('save {} segmented data in {}'.format(num_speech, save_path))
		for ind in range(num_speech):
			filePath = save_path+'_segment-'+str(ind)+".npz"
			np.savez(filePath,
			 		 mix_speech=mix_speech[ind,:,:],
			 		 speech1=speech1[ind,:,:],
			 		 speech2=speech2[ind,:,:],
					 snr=snr)
			testData = np.load(filePath, allow_pickle=True)
			testData.close()
				

	def data_generator(self):
		train_data_dir = os.path.join(self.dataset_base_dir+self.clean_speech_dir, 'train','files')
		train_npz_dir = os.path.join(self.dataset_base_dir+self.clean_speech_dir, 'train','npz')
		train_mix_data_dir = os.path.join(self.dataset_base_dir+self.mixed_files_save_path, 'train','files')
		train_mix_npz_dir = os.path.join(self.dataset_base_dir+self.mixed_files_save_path, 'train','npz',str(self.len_time))
		validation_data_dir = os.path.join(self.dataset_base_dir+self.clean_speech_dir, 'validation','files')
		validation_npz_dir = os.path.join(self.dataset_base_dir+self.clean_speech_dir, 'validation','npz')
		validation_mix_dir = os.path.join(self.dataset_base_dir+self.mixed_files_save_path, 'validation','files')
		validation_mix_npz_dir = os.path.join(self.dataset_base_dir+self.mixed_files_save_path, 'validation','npz',str(self.len_time))
		
		Path(train_data_dir).mkdir(parents=True, exist_ok=True)
		Path(train_npz_dir).mkdir(parents=True, exist_ok=True)
		Path(train_mix_data_dir).mkdir(parents=True, exist_ok=True)
		Path(train_mix_npz_dir).mkdir(parents=True, exist_ok=True)
		Path(validation_data_dir).mkdir(parents=True, exist_ok=True)
		Path(validation_npz_dir).mkdir(parents=True, exist_ok=True)
		Path(validation_mix_dir).mkdir(parents=True, exist_ok=True)
		Path(validation_mix_npz_dir).mkdir(parents=True, exist_ok=True)

		print('DATA GENERATOR & MIXER')
		print('train data dir: '+train_data_dir)
		print('train npz dir: '+train_npz_dir)
		print('train mix data dir: '+train_mix_data_dir)
		print('train mix npz dir: '+train_mix_npz_dir)
		print('validation data dir: '+validation_data_dir)
		print('validation npz dir: '+validation_npz_dir)
		print('validation mix data dir: '+validation_mix_dir)
		print('validation mix npz dir: '+validation_mix_npz_dir)


		print('#####generate train_data######')
		trainNpzfile = np.load(train_npz_dir+'/clean-0.5.npz', allow_pickle=True)
		print(trainNpzfile.files)
		print("records count: {}".format(len(trainNpzfile['data'])))

		data = trainNpzfile['data']
		
		counter = 1
		generatedFiles = 0

		for i,(speaker1,audio1,data1) in enumerate(data):
			for j,(speaker2,audio2,data2) in enumerate(data[counter:], counter):
				if speaker2 == speaker1 :
					# print("skip")
					continue
				
				# input_path = os.path.join(train_mix_npz_dir,'{}-{}_{}-{}'.format(speaker1,audio1[:-4],speaker2,audio2[:-4]))
				# if Path(input_path+'_segment-0.npz').is_file():
				# 	print('skip exist file')
				# 	continue

				randomSNR = randrange(6)
				print('mix {}/{} & {}/{} with SNR {}'.format(speaker1,audio1,speaker2,audio2,randomSNR))
				
				mixSpeech, speech1, speech2 = mix(data1,data2,randomSNR)
				
				# sample to create mixture file
				if generatedFiles % 300 == 0:
					wavfile.write('{}/{}-{}_{}-{}.wav'.format(train_mix_data_dir,speaker1,audio1[:-4],speaker2,audio2[:-4]), 8000, np.asarray(mixSpeech))
				
				mixSpeech, speech1, speech2 = self.segment_audio(mixSpeech, speech1, speech2)

				# input_path = os.path.join(train_mix_npz_dir,'{}-{}_{}-{}'.format(speaker1,audio1[:-4],speaker2,audio2[:-4]))
				filename = '{}-{}_{}-{}'.format(speaker1,audio1[:-4],speaker2,audio2[:-4])
				self.save_speech(mixSpeech,speech1,speech2,randomSNR,filename)

				generatedFiles += 1
			counter+=1
			

		print('{} mixed files generated'.format(generatedFiles))
		print('{} train set and {} validation set'.format(self.trainSet, self.validSet))
		trainNpzfile.close()


		# print('#####generate test_data######')
		# validationNpzfile = np.load(validation_npz_dir+'/clean-0.5.npz', allow_pickle=True)
		# print("records count: {}".format(len(validationNpzfile['data'])))

		# data = validationNpzfile['data']
		
		# counter = 1
		# generatedFiles = 0

		# for i,(speaker1,audio1,data1) in enumerate(data):
		# 	for j,(speaker2,audio2,data2) in enumerate(data[counter:], counter):
		# 		if speaker2 == speaker1 :
		# 			# print("skip")
		# 			continue
				
		# 		randomSNR = randrange(6)
		# 		print('mix {}/{} & {}/{} with SNR {}'.format(speaker1,audio1,speaker2,audio2,randomSNR))
				
		# 		mixSpeech, speech1, speech2 = mix(data1,data2,randomSNR)
				
		# 		# sample to create mixture file
		# 		if generatedFiles % 100 == 0:
		# 			wavfile.write('{}/{}-{}_{}-{}.wav'.format(validation_mix_dir,speaker1,audio1[:-4],speaker2,audio2[:-4]), 8000, np.asarray(mixSpeech))

		# 		mixSpeech, speech1, speech2 = self.segment_audio(mixSpeech, speech1, speech2)
					
		# 		input_path = os.path.join(validation_mix_npz_dir,'{}-{}_{}-{}'.format(speaker1,audio1[:-4],speaker2,audio2[:-4]))
		# 		self.save_test_speech(mixSpeech,speech1,speech2,randomSNR,input_path)

		# 		generatedFiles += 1
		# 	counter+=1

		# validationNpzfile.close()

		
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="TasNet by PyTorch ")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        dest="config",
        help="Location of .yaml configure files for training")
    args = parser.parse_args()
    config_dict = parse_yaml(args.config)
    data_config = config_dict["data_generator"]
    processor = speech_preprocess(**data_config)
    processor.data_generator()
