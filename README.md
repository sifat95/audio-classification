# audio-classification

```shell
# Upgrade pip first. Also make sure wheel is installed.
python -m pip install --upgrade pip wheel

# Install dependences
pip install -r requirements.txt
# faced errors installing pyaudio with pip, conda installation worked fine

# Download model file
curl -O https://storage.googleapis.com/audioset/yamnet.h5

# create a directory for recorded audio
mkdir recorded_audio

# Clone TensorFlow yamnet model repo 
git clone https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

# Installation ready, let's test it. It will start a recorder and show predicted audio class on the console
python classify_from_mic.py
# device index may need to be adjusted, default is 0

```
# Reference
 
1. https://github.com/laanlabs/train_detector

2. https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
