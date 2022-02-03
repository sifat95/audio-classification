#!/usr/bin/env python
# coding: utf-8

import os, sys, uuid

import numpy as np
import pandas as pd

import tensorflow as tf
# import tensorflow_hub as hub
# import tensorflow_io as tfio

yamnet_base = './YAMNet_transfer'
sys.path.append(yamnet_base)

import os

# assert os.path.exists(yamnet_base)

import time

# audio stuff 
import librosa
import soundfile as sf
import resampy
import pyaudio
import params as yamnet_params
import yamnet as yamnet_model
# yamnet imports 
# import params
# import modified_yamnet as yamnet_model
# import features as features_lib

# TF / keras 
#from tensorflow.keras import Model, layers
# import tensorflow as tf
# from tensorflow.keras.models import load_model


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


DESIRED_SR = 16000


# TODO: include my slightly modified yamnet code in this file 
# i added the 'dense_net' return
# def load_yamnet_model(model_path='yamnet.h5'):
#     # Set up the YAMNet model.
#     params.PATCH_HOP_SECONDS = 0.1  # 10 Hz scores frame rate.
#     yamnet, dense_net = yamnet_model.yamnet_frames_model(params)
#     yamnet.load_weights(model_path)
#     return yamnet


# def load_top_model(model_path="top_model.h5"):
#     return load_model(model_path)


def read_wav(fname, output_sr, use_rosa=False):
    
    if use_rosa:
        waveform, sr = librosa.load(fname, sr=output_sr)
    else:
        wav_data, sr = sf.read(fname, dtype=np.int16)
        
        if wav_data.ndim > 1: 
            # (ns, 2)
            wav_data = wav_data.mean(1)
        if sr != output_sr:
            wav_data = resampy.resample(wav_data, sr, output_sr)
        waveform = wav_data / 32768.0
    
    return waveform.astype(np.float64)


def remove_silence(waveform, top_db=15, min_chunk_size=2000, merge_chunks=True):
    """
    Loads sample into chunks of non-silence 
    """
    splits = librosa.effects.split(waveform, top_db=top_db)
    
    waves = []
    for start, end in splits:
        if (end-start) < min_chunk_size:
            continue
        waves.append(waveform[start:end])
    
    if merge_chunks and len(waves) > 0:
        waves = np.concatenate(waves)
    
    return waves


def run_models(waveform, 
               yamnet_model, 
               top_model, 
               strip_silence=True, 
               min_samples=11000):
    
    if strip_silence:
        waveform = remove_silence(waveform, top_db=10)
    
    if waveform is None:
        print('none wav?')
        return None
    
    if len(waveform) < min_samples:
        #print(" too small after silence: " , len(waveform))
        return None
    
    # predictions, spectrogram, net, patches
    _scores, _spectro, dense_out, _patches = \
        yamnet_model.predict(np.reshape(waveform, [1, -1]), steps=1)
    

    # dense = (N, 1024)
    all_scores = []
    for patch in dense_out:
        scores = top_model.predict( np.expand_dims(patch, 0) ).squeeze()
        all_scores.append(scores)
    
    if not all_scores:
        # no patches returned
        return None
    
    all_scores = np.mean(all_scores, axis=0)
    return all_scores



def run_detection_loop(input_device_index = 0):

    # yamnet = load_yamnet_model()
    # top_model = load_top_model()
    # yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    # yamnet_model = hub.load(yamnet_model_handle)
    # class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    # class_names =list(pd.read_csv(class_map_path)['display_name'])


    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    yamnet_classes = yamnet_model.class_names('./YAMNet_transfer/yamnet_class_map.csv')

    CHUNK = 4096 * 2

    FORMAT = pyaudio.paInt16

    DTYPE = np.int16 if FORMAT == pyaudio.paInt16 else np.float32

    CHANNELS = 1
    RATE = 44100


    min_frames_to_process = int(DESIRED_SR * 5.5)

    p = pyaudio.PyAudio()
    p.get_device_count()

    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i))
        print("_______")

    p.terminate()


    def dump_wav(arr, fname):    
        # librosa.output.write_wav(fname, arr, DESIRED_SR)
        sf.write(fname, arr, DESIRED_SR, 'PCM_16')


    if not os.path.exists("logs"):
        os.makedirs("logs")

    if not os.path.exists("detections"):
        os.makedirs("detections")        

    if not os.path.exists("train_wavs"):
        os.makedirs("train_wavs")
        os.makedirs("train_wavs/high")
        os.makedirs("train_wavs/mid")
        os.makedirs("train_wavs/low")

    def log_line(line, type='info'):
        timestr = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime() )
        log_file.write("{:<30} [{:<8}] {}\n".format(timestr, type, line))
        log_file.flush()

    def log_detection(score, wav_file=""):
        timestr = time.strftime('%a %d %b %Y %H:%M:%S', time.localtime() )
        timestr2 = str(time.time())
        score = np.round(score, 3)
        csv_file.write("{},{},{},{}\n".format(timestr, timestr2, score, wav_file))
        csv_file.flush()

    # Utility functions for loading audio files and making sure the sample rate is correct.

    @tf.function
    def load_wav_16k_mono(file_contents):
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        # file_contents = tf.io.read_file(filename)
        # wav, sample_rate = tf.audio.decode_wav(
        #     file_contents,
        #     desired_channels=1)

        # wav = tf.squeeze(file_contents, axis=-1)
        sample_rate = tf.cast(DESIRED_SR, dtype=tf.int64)
        wav = tfio.audio.resample(file_contents, rate_in=sample_rate, rate_out=16000)
        
        return wav
  

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK)

    frames = []
    chunks_required = int(np.ceil(min_frames_to_process // CHUNK))

    MIN_NOISE = 0.1
    NOISE_MEAN_SCALE = 30.0

    top_db = 18

    MIN_SAMPLES_TO_RUN_NN = 5500


    verbose = 0

    timestr = time.strftime('%a_%d_%b_%Y_%H-%M-%S', time.localtime())

    log_name = "logs/{}.txt".format(timestr)
    log_file = open(log_name, 'w')

    csv_name = "detections/{}.csv".format(timestr)
    csv_file = open(csv_name, 'w', encoding='utf-8')

    last_web_update = time.time()
    last_ping_time = time.time()


    try:
        
        while True:
            
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except OSError:
                print(" __ overflow")

            arr = np.frombuffer(data, dtype=DTYPE)
            # arr = load_wav_16k_mono(arr)
            # arr = tf.cast(arr, tf.float32)

            arr = arr.astype(np.float32)
            arr = arr / 32768.0
            # arr = arr * 1.25 
            if len(arr.shape) > 1:
                arr = np.mean(arr, axis=1)
            if RATE != DESIRED_SR:
                 arr = resampy.resample(arr, RATE, DESIRED_SR)
            frames.append(arr)
            
            if len(frames) > chunks_required:
                frames.pop(0)
            
            if len(frames) >= chunks_required:
                
                wave_arr = np.concatenate(frames)
                noise_mean = np.abs(wave_arr).mean() * NOISE_MEAN_SCALE
                
                if noise_mean < MIN_NOISE:
                    continue
                
                # wave_arr = remove_silence(wave_arr, top_db=top_db)
                
                if wave_arr is None:
                    continue 
                    
                # hack .. double the wave if too short
                if len(wave_arr) > MIN_SAMPLES_TO_RUN_NN//2 and len(wave_arr) < MIN_SAMPLES_TO_RUN_NN:
                    wave_arr = np.concatenate((wave_arr, wave_arr))
                
                scores = None
                
                noise_mean = np.abs(wave_arr).mean() * NOISE_MEAN_SCALE
                
                if noise_mean < MIN_NOISE:
                    continue
                # wave_arr = load_wav_16k_mono(wave_arr)
                
                if wave_arr is not None and len(wave_arr) >= MIN_SAMPLES_TO_RUN_NN:
                    scores, embeddings, spectrogram = yamnet(wave_arr)
                    class_scores = tf.reduce_mean(scores, axis=0)
                    top_class = tf.argmax(class_scores)
                    inferred_class = yamnet_classes[top_class]

                    print('The main sound is: {}', inferred_class)
                    print('The embeddings shape: {}', embeddings.shape)
                        
                    file_name = '{}.wav'.format(uuid.uuid4().hex)
                    wav_out_path = os.path.join('./recorded_audio', file_name)
                    dump_wav(wave_arr, wav_out_path)


                
    except KeyboardInterrupt as e:
        print(" ____ interrupt ___")
        stream.stop_stream()
        stream.close()
        p.terminate()
        log_file.close()
        
    except Exception as e:
        stream.stop_stream()
        stream.close()
        p.terminate()
        log_file.close()
        print(" err" , str(e))
        raise e


if __name__ == '__main__':
    input_device_index = 18
    print(sys.argv)
    if len(sys.argv) > 1:
        input_device_index = int(sys.argv[1])
    print(" --- Using input device: ", input_device_index)
    run_detection_loop(input_device_index=input_device_index)



