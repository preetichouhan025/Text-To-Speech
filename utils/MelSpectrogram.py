
from torchaudio import transforms
import torch
import torch.nn.functional as F


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        hop_length,
        win_length,
        n_fft,
        n_mels,
        f_min,
        f_max,
        power,
        normalized,
        norm,
        mel_scale,
        compression
    ):
        """
        MelSpectrogram computes the Mel spectrogram of an audio waveform.

        Args:
            sample_rate (int): The sample rate of the audio waveform.
            hop_length (int): The hop length for computing the spectrogram.
            win_length (int): The window length for computing the spectrogram.
            n_fft (int): The size of the FFT window.
            n_mels (int): The number of Mel bands to generate.
            f_min (float): The minimum frequency for the Mel filter bank.
            f_max (float): The maximum frequency for the Mel filter bank.
            power (float): The exponent for the magnitude spectrogram.
            normalized (bool): Whether to normalize the final Mel spectrogram.
            norm (float, optional): The value to normalize the final Mel spectrogram.
                                    Defaults to None.
            mel_scale (str): The scale to use for the Mel filter bank.
                             Should be either "linear" or "log".
            compression (bool): Whether to apply dynamic range compression to the spectrogram.


        Example:

        # Create an instance of MelSpectrogram
        mel_spectrogram = MelSpectrogram(
            sample_rate=22050,
            hop_length=512,
            win_length=1024,
            n_fft=2048,
            n_mels=128,
            f_min=0.0,
            f_max=8000.0,
            power=2.0,
            normalized=True,
            norm=None,
            mel_scale='linear',
            compression=True
        )

        # Generate a random input audio tensor
        audio = torch.randn(1, 44100)

        # Compute the Mel spectrogram of the input audio
        mel_spec = mel_spectrogram(audio)

        print(mel_spec.shape)  # Output: torch.Size([1, 128, 87])

        This code sample is taken from Speechbrain implementation

        """

        super(MelSpectrogram, self).__init__()

        self.audio_to_mel = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
            normalized=normalized,
            norm=norm,
            mel_scale=mel_scale
        )

        self.compression = compression

    def __call__(self, audio):
        """
        Compute the Mel spectrogram of the input audio.

        Args:
            audio (torch.Tensor): The input audio waveform.

        Returns:
            torch.Tensor: The computed Mel spectrogram.
        """

        mel = self.audio_to_mel(audio)

        if self.compression:
            mel = MelSpectrogram.dynamic_range_compression(mel)

        return mel

    @staticmethod
    def dynamic_range_compression(x, C=1, clip_val=1e-5):


        """
        Apply dynamic range compression to the input spectrogram.

        Args:
            x (torch.Tensor): The input spectrogram.
            C (float, optional): Compression factor. Defaults to 1.
            clip_val (float, optional): Value to clip the input. Defaults to 1e-5.

        Returns:
            torch.Tensor: The compressed spectrogram.
        """

        return torch.log(torch.clamp(x, min=clip_val) * C)



