## Potential model 1: BirdNet

### Reference
BirdNET: A deep learning solution for avian diversity monitoring ([link](https://www.sciencedirect.com/science/article/pii/S1574954121000273#:~:text=In%20summary%2C%20BirdNET%20achieved%20a,4%20years%20of%20audio%20data.))

### Key data processing steps
1. Spectrogram with a FFT window size of 10.7ms (512 samples at 48kHz sampling rate), overlap of 25%, 8ms per frame. Frequency range is restricted from 150Hz to 15kHz.
      - Frequency compression using the mel scale with 64 bands, break frequency at 1750 Hz (linear scaling up to 500 Hz)
2. Duration of bird vocalization is set to 3s chunks
3. Signal strength estimation to determine presence of a bird sound (see 2016 BirdCLEF edition)

### Data augmentation
1. Random shifts in frequency and time (vertical and horizontal roll)
2. Random partial stretching in time and frequency (warping)
3. Addtion of noise from samples with non-salient chunks of audio

### Model architecture
1. Wide ResNet
2. Trained using 1.5 million spectrograms with 3500 samples per class. Oversampling was done to reduce class imbalance
3. Use mixup training
