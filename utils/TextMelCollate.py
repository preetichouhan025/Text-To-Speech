
from torchaudio import transforms
import torch
import torch.nn.functional as F


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step

    Arguments
    ---------
    n_frames_per_step: int
        the number of output frames per step

    Returns
    -------
    result: tuple
        A tuple of tensors to be used as inputs/targets
        (
            text_padded: torch.Tensor
                Padded tensor of normalized text.
            input_lengths: torch.Tensor
                Tensor containing the length of each text input.
            mel_bos_padded: torch.Tensor
                Padded tensor of mel-spectrograms with the beginning-of-sentence (BOS) token.
            mel_eos_padded: torch.Tensor
                Padded tensor of mel-spectrograms with the end-of-sentence (EOS) token.
            output_lengths: torch.Tensor
                Tensor containing the length of each mel-spectrogram output.
            len_encoded_phoeneme: torch.Tensor
                Tensor containing the length of each encoded phoneme sequence.
            labels: list
                List containing labels for each batch element.
            wavs: list
                List containing waveforms for each batch element.
            mel_specs: list
                List containing mel-spectrograms for each batch element.
            start_tokens: list
                List containing start tokens for each batch element.
            stop_tokens: list
                List containing stop tokens for each batch element.
        )

    Example:
    --------
    >>> collate_fn = TextMelCollate(n_frames_per_step=5)
    >>> batch = [
    ...     {"mel_text_pair": (text_normalized_1, mel_normalized_1), "label": label_1, "wav": wav_1},
    ...     {"mel_text_pair": (text_normalized_2, mel_normalized_2), "label": label_2, "wav": wav_2},
    ...     ...
    ... ]
    >>> text_padded, input_lengths, mel_padded_bos, mel_padded_eos, output_lengths, len_encoded_phoeneme, labels, wavs, mel_specs, start_tokens, stop_tokens = collate_fn(batch)
    """

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step


    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        Arguments
        ---------
        batch: list
            [text_normalized, mel_normalized]
        """


        raw_batch = list(batch)
        for i in range(
            len(batch)
        ):  # the pipline return a dictionary with one elemnent
            batch[i] = batch[i]["mel_text_pair"]

        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text


        max_target_len = max([x[5].size(1) for x in batch])
        # get stop/EOS Token
        stop_token = torch.full((1, max_target_len), fill_value= 0)
        stop_token[-1] = 1 # Set the last element of the EOS tensor to 1




        num_mels = batch[0][5].size(0)

        # Right zero-pad mel_spec_bos
        mel_padded_bos = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded_bos.zero_()

        # Right zero-pad mel_spec_eos
        mel_padded_eos = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded_eos.zero_()


        output_lengths = torch.LongTensor(len(batch))
        labels, wavs = [], []
        mel_specs, stop_tokens, start_tokens = [], [], []


        for i in range(len(ids_sorted_decreasing)):
            idx = ids_sorted_decreasing[i]

            # Right zero-pad mel_spec_bos
            mel_spec_bos = batch[idx][5]
            mel_padded_bos[i, :, : mel_spec_bos.size(1)] = mel_spec_bos

            # Right zero-pad mel_spec_eos
            mel_spec_eos = batch[idx][6]
            mel_padded_eos[i, :, : mel_spec_eos.size(1)] = mel_spec_eos

            output_lengths[i] = mel_padded_bos.size(1)


            labels.append(raw_batch[idx]["label"])
            wavs.append(raw_batch[idx]["wav"])
            stop_tokens.append(stop_token)
            mel_specs.append(raw_batch[idx]["mel_text_pair"][1])
            start_tokens.append(raw_batch[idx]["mel_text_pair"][4])


        # count number of items - characters in text
        len_encoded_phoeneme = [x[2] for x in batch]
        len_encoded_phoeneme = torch.Tensor(len_encoded_phoeneme)

        return (
            text_padded,
            input_lengths,
            mel_padded_bos,
            mel_padded_eos,
            output_lengths,
            len_encoded_phoeneme,
            labels,
            wavs,
            mel_specs,
            start_tokens,
            stop_tokens
        )


