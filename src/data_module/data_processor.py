"""Data processing modules"""
import glob
import os
from typing import List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class BirdPresenceDetector:
    @staticmethod
    def create_mel_filterbank(
        sample_rate, frame_len, num_bands, min_freq, max_freq, corner=700
    ):
        """
        Creates a mel filterbank of `num_bands` triangular filters, with the first
        filter starting at `min_freq` and the last one stopping at `max_freq`.
        Returns the filterbank as a matrix suitable for a dot product against
        magnitude spectra created from samples at a sample rate of `sample_rate`
        with a window length of `frame_len` samples.
        """
        # prepare output matrix
        input_bins = (frame_len // 2) + 1
        filterbank = np.zeros((input_bins, num_bands))

        # mel-spaced peak frequencies
        coefficient = 1000 / np.log1p(1000 / corner)
        min_mel = coefficient * np.log1p(min_freq / corner)
        max_mel = coefficient * np.log1p(max_freq / corner)
        spacing = (max_mel - min_mel) / (num_bands + 1)
        peaks_mel = min_mel + np.arange(num_bands + 2) * spacing
        peaks_hz = corner * (np.exp(peaks_mel / coefficient) - 1)
        fft_freqs = np.linspace(0, sample_rate / 2.0, input_bins)
        peaks_bin = np.searchsorted(fft_freqs, peaks_hz)

        # fill output matrix with triangular filters
        for b, filt in enumerate(filterbank.T):
            # The triangle starts at the previous filter's peak (peaks_freq[b]),
            # has its maximum at peaks_freq[b+1] and ends at peaks_freq[b+2].
            left_hz, _, right_hz = peaks_hz[b : b + 3]  # b, b+1, b+2
            left_bin, top_bin, right_bin = peaks_bin[b : b + 3]
            # Create triangular filter compatible to yaafe
            filt[left_bin:top_bin] = (fft_freqs[left_bin:top_bin] - left_hz) / (
                top_bin - left_bin
            )
            filt[top_bin:right_bin] = (right_hz - fft_freqs[top_bin:right_bin]) / (
                right_bin - top_bin
            )
            filt[left_bin:right_bin] /= filt[left_bin:right_bin].sum()

        return filterbank

    @staticmethod
    def calculate_indicator(melspec, factor=3):
        # calculate signal regions
        row_median = np.median(melspec, axis=1).reshape(-1, 1)
        col_median = np.median(melspec, axis=0).reshape(1, -1)

        row_mask = melspec > (row_median * factor)
        col_mask = melspec > (col_median * factor)
        mask = row_mask * col_mask

        # perform erosion then dilation
        mask = skimage.morphology.binary_erosion(mask, footprint=np.ones((4, 4)))
        mask = skimage.morphology.binary_dilation(mask, footprint=np.ones((4, 4)))

        indicator = np.any(mask, axis=0)
        indicator = skimage.morphology.binary_dilation(
            indicator, footprint=np.ones((4,))
        )
        indicator = skimage.morphology.binary_dilation(
            indicator, footprint=np.ones((4,))
        )

        return indicator

    @staticmethod
    def concatenate_parts(indicator, y):
        change = np.where(indicator[:-1] != indicator[1:])[0]

        if indicator[0] == True:
            left_edge = np.concatenate(([0], change[1::2]))
            right_edge = change[0::2]
        else:
            left_edge = change[::2]
            right_edge = change[1::2]

        if indicator[-1] == True:
            right_edge = np.append(right_edge, len(indicator))

        left_timings = left_edge * 512 * 0.75 - 256
        left_timings = left_timings.astype(int)
        right_timings = right_edge * 512 * 0.75 - 256
        right_timings = right_timings.astype(int)
        parts = [y[j : k + 1] for (j, k) in zip(left_timings, right_timings)]
        if len(parts) > 0:
            output_y = np.concatenate(parts)
            return output_y

    @classmethod
    def separate_signal_from_noise(cls, y, sr):
        x_hann = librosa.stft(
            y, window="hann", n_fft=512, win_length=512, hop_length=int(512 * 0.75)
        )
        melfb = cls.create_mel_filterbank(
            sample_rate=sr,
            frame_len=512,
            num_bands=64,
            min_freq=150,
            max_freq=15000,
            corner=1750,
        ).T
        melspec: np.ndarray = np.einsum(
            "...ft,mf->...mt", np.abs(x_hann), melfb, optimize=True
        )

        signal_indicator = cls.calculate_indicator(melspec, factor=3)

        noise_indicator = cls.calculate_indicator(melspec, factor=2.5)
        noise_indicator = ~noise_indicator

        signal_y = cls.concatenate_parts(signal_indicator, y)
        noise_y = cls.concatenate_parts(noise_indicator, y)

        return signal_y, noise_y


class DataPipeline:
    def __init__(self, audio_files) -> None:
        self.bird_presence_detector = BirdPresenceDetector()
        self.audio_files = audio_files

    @staticmethod
    def load_audio(filepath: str, sample_rate=48000) -> Tuple[np.ndarray, int]:
        return librosa.load(filepath, sr=sample_rate, mono=True)

    @staticmethod
    def write_output(folder, filename, output):
        filepaths = []
        for i in range(len(output)):
            filepath = os.path.join(folder, f"{filename}_{i}.tif")
            skimage.io.imsave(filepath, output[i])
            filepaths.append(filepath)

        return filepaths

    def split_audio(self, y, length=384, sr=48000, overlap=0):
        if len(y) < 512:
            return []
        x_hann = librosa.stft(
            y, window="hann", n_fft=512, win_length=512, hop_length=int(512 * 0.75)
        )
        melfb = self.bird_presence_detector.create_mel_filterbank(
            sample_rate=sr,
            frame_len=512,
            num_bands=64,
            min_freq=150,
            max_freq=15000,
            corner=1750,
        ).T
        melspec: np.ndarray = np.einsum(
            "...ft,mf->...mt", np.abs(x_hann), melfb, optimize=True
        )

        output = []
        n_samples = int((melspec.shape[1]) // (length - overlap * length))
        for i in range(n_samples):
            if int(i * (length - overlap * length)) + length > melspec.shape[1]:
                # ensure that no spectrogram is truncated
                break
            output.append(
                melspec[
                    :,
                    int(i * (length - overlap * length)) : int(
                        i * (length - overlap * length)
                    )
                    + length,
                ]
            )

        return output

    def train_val_test_split(self, df, val_split=0.1, test_split=0.1):
        train, val_test = train_test_split(
            df, test_size=val_split + test_split, random_state=42, stratify=df.label
        )
        val, test = train_test_split(
            val_test,
            test_size=(val_split) / (val_split + test_split),
            random_state=42,
            stratify=val_test.label,
        )

        return train, val, test

    def run(self):
        output_df = pd.DataFrame(columns=["filepath", "label"])
        omit_count = 0
        for file in tqdm(self.audio_files):
            folder, filename = os.path.split(file)
            filename = os.path.splitext(filename)[0]
            animal = os.path.split(folder)[-1]
            split_folder = os.path.realpath(
                os.path.join(folder, "..", "..", "audio_split")
            )
            os.makedirs(split_folder, exist_ok=True)
            signal_folder = os.path.join(split_folder, "signal")
            os.makedirs(signal_folder, exist_ok=True)
            noise_folder = os.path.join(split_folder, "noise")
            os.makedirs(noise_folder, exist_ok=True)

            y, sr = self.load_audio(file)
            signal_y, noise_y = self.bird_presence_detector.separate_signal_from_noise(
                y, sr
            )

            if signal_y is not None:
                signal_output = self.split_audio(signal_y, sr=sr)
                filepaths = self.write_output(signal_folder, filename, signal_output)
                output_df = pd.concat(
                    (
                        output_df,
                        pd.DataFrame.from_dict(
                            {
                                "filepath": filepaths,
                                "label": [animal] * len(filepaths),
                            }
                        ),
                    )
                )

            if noise_y is not None:
                noise_output = self.split_audio(noise_y, sr=sr)
                filepaths = self.write_output(noise_folder, filename, noise_output)
                output_df = pd.concat(
                    (
                        output_df,
                        pd.DataFrame.from_dict(
                            {
                                "filepath": filepaths,
                                "label": ["noise"] * len(filepaths),
                            }
                        ),
                    )
                )

        print(omit_count)
        output_df = output_df.reset_index(drop=True)

        train, val, test = self.train_val_test_split(output_df)
        train.to_csv(os.path.join(split_folder, "train.csv"), index=False)
        val.to_csv(os.path.join(split_folder, "val.csv"), index=False)
        test.to_csv(os.path.join(split_folder, "test.csv"), index=False)


if __name__ == "__main__":
    audio_folder = "./data/dataset/audio"
    audio_files = glob.glob(os.path.join(audio_folder, "**", "*.mp3"))

    pipeline = DataPipeline(audio_files)
    pipeline.run()
