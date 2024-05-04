
import re
import logging
import torch
import torchaudio
import random
import speechbrain
from speechbrain.inference.interfaces import Pretrained
from speechbrain.inference.text import GraphemeToPhoneme

logger = logging.getLogger(__name__)

class TTSInferencing(Pretrained):
    """
    A ready-to-use wrapper for TTS (text -> mel_spec).
    Arguments
    ---------
    hparams
        Hyperparameters (from HyperPyYAML)
    """

    HPARAMS_NEEDED = ["modules", "input_encoder"]

    MODULES_NEEDED = ["encoder_prenet", "pos_emb_enc",
                      "decoder_prenet", "pos_emb_dec",
                      "Seq2SeqTransformer", "mel_lin",
                      "stop_lin", "decoder_postnet"]


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lexicon = self.hparams.lexicon
        lexicon = ["@@"] + lexicon
        self.input_encoder = self.hparams.input_encoder
        self.input_encoder.update_from_iterable(lexicon, sequence_input=False)
        self.input_encoder.add_unk()

        self.modules = self.hparams.modules

        self.g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p")




    def generate_padded_phonemes(self, texts):
        """Computes mel-spectrogram for a list of texts

        Arguments
        ---------
        texts: List[str]
            texts to be converted to spectrogram

        Returns
        -------
        tensors of output spectrograms
        """

        # Preprocessing required at the inference time for the input text
        # "label" below contains input text
        # "phoneme_labels" contain the phoneme sequences corresponding to input text labels

        phoneme_labels = list()

        for label in texts:

          phoneme_label = list()

          label = self.custom_clean(label).upper()

          words = label.split()
          words = [word.strip() for word in words]
          words_phonemes = self.g2p(words)

          for i in range(len(words_phonemes)):
              words_phonemes_seq = words_phonemes[i]
              for phoneme in words_phonemes_seq:
                  if not phoneme.isspace():
                      phoneme_label.append(phoneme)
          phoneme_labels.append(phoneme_label)


        # encode the phonemes with input text encoder
        encoded_phonemes = list()
        for i in range(len(phoneme_labels)):
            phoneme_label = phoneme_labels[i]
            encoded_phoneme =  torch.LongTensor(self.input_encoder.encode_sequence(phoneme_label)).to(self.device)
            encoded_phonemes.append(encoded_phoneme)


        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x) for x in encoded_phonemes]), dim=0, descending=True
        )

        max_input_len = input_lengths[0]

        phoneme_padded = torch.LongTensor(len(encoded_phonemes), max_input_len).to(self.device)
        phoneme_padded.zero_()

        for seq_idx, seq in enumerate(encoded_phonemes):
            phoneme_padded[seq_idx, : len(seq)] = seq


        return phoneme_padded.to(self.device, non_blocking=True).float()


    def encode_batch(self, texts):
        """Computes mel-spectrogram for a list of texts

        Texts must be sorted in decreasing order on their lengths

        Arguments
        ---------
        texts: List[str]
            texts to be encoded into spectrogram

        Returns
        -------
        tensors of output spectrograms
        """

        # generate phonemes and padd the input texts
        encoded_phoneme_padded = self.generate_padded_phonemes(texts)
        phoneme_prenet_emb = self.modules['encoder_prenet'](encoded_phoneme_padded)
        # Positional Embeddings
        phoneme_pos_emb =  self.modules['pos_emb_enc'](encoded_phoneme_padded)
        # Summing up embeddings
        enc_phoneme_emb = phoneme_prenet_emb.permute(0,2,1)  + phoneme_pos_emb
        enc_phoneme_emb = enc_phoneme_emb.to(self.device)


        with torch.no_grad():

          # generate sequential predictions via transformer decoder
          start_token = torch.full((80, 1), fill_value= 0)
          start_token[1] = 2
          decoder_input = start_token.repeat(enc_phoneme_emb.size(0), 1, 1)
          decoder_input = decoder_input.to(self.device, non_blocking=True).float()

          num_itr = 0
          stop_condition = [False] * decoder_input.size(0)
          max_iter = 100

          # while not all(stop_condition) and num_itr < max_iter:
          while num_itr < max_iter:

            # Decoder Prenet
            mel_prenet_emb =  self.modules['decoder_prenet'](decoder_input).to(self.device).permute(0,2,1)

            # Positional Embeddings
            mel_pos_emb =  self.modules['pos_emb_dec'](mel_prenet_emb).to(self.device)
            # Summing up Embeddings
            dec_mel_spec = mel_prenet_emb + mel_pos_emb

            # Getting the target mask to avoid looking ahead
            tgt_mask = self.hparams.lookahead_mask(dec_mel_spec).to(self.device)

            # Getting the source mask
            src_mask = torch.zeros(enc_phoneme_emb.shape[1], enc_phoneme_emb.shape[1]).to(self.device)

            # Padding masks for source and targets
            src_key_padding_mask = self.hparams.padding_mask(enc_phoneme_emb, pad_idx = self.hparams.blank_index).to(self.device)
            tgt_key_padding_mask = self.hparams.padding_mask(dec_mel_spec, pad_idx = self.hparams.blank_index).to(self.device)


            # Running the Seq2Seq Transformer
            decoder_outputs = self.modules['Seq2SeqTransformer'](src = enc_phoneme_emb, tgt = dec_mel_spec, src_mask = src_mask, tgt_mask = tgt_mask,
                                                              src_key_padding_mask = src_key_padding_mask, tgt_key_padding_mask = tgt_key_padding_mask)

            # Mel Linears
            mel_linears =  self.modules['mel_lin'](decoder_outputs).permute(0,2,1)
            mel_postnet = self.modules['decoder_postnet'](mel_linears) # mel tensor output
            mel_pred = mel_linears + mel_postnet # mel tensor output

            stop_token_pred =  self.modules['stop_lin'](decoder_outputs).squeeze(-1)

            stop_condition_list = self.check_stop_condition(stop_token_pred)


            # update the values of main stop conditions
            stop_condition_update = [True if stop_condition_list[i] else stop_condition[i] for i in range(len(stop_condition))]
            stop_condition = stop_condition_update


            # Prepare input for the transformer input for next iteration
            current_output = mel_pred[:, :, -1:]

            decoder_input=torch.cat([decoder_input,current_output],dim=2)
            num_itr = num_itr+1

        mel_outputs =  decoder_input[:, :, 1:]

        return mel_outputs



    def encode_text(self, text):
        """Runs inference for a single text str"""
        return self.encode_batch([text])


    def forward(self, text_list):
        "Encodes the input texts."
        return self.encode_batch(text_list)


    def check_stop_condition(self, stop_token_pred):
        """
        check if stop token / EOS reached or not for mel_specs in the batch
        """

        # Applying sigmoid to perform binary classification
        sigmoid_output = torch.sigmoid(stop_token_pred)
        # Checking if the probability is greater than 0.5
        stop_results = sigmoid_output > 0.8
        stop_output = [all(result) for result in stop_results]

        return stop_output



    def custom_clean(self, text):
        """
        Uses custom criteria to clean text.

        Arguments
        ---------
        text : str
            Input text to be cleaned
        model_name : str
            whether to treat punctuations

        Returns
        -------
        text : str
            Cleaned text
        """

        _abbreviations = [
            (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
            for x in [
                ("mrs", "missus"),
                ("mr", "mister"),
                ("dr", "doctor"),
                ("st", "saint"),
                ("co", "company"),
                ("jr", "junior"),
                ("maj", "major"),
                ("gen", "general"),
                ("drs", "doctors"),
                ("rev", "reverend"),
                ("lt", "lieutenant"),
                ("hon", "honorable"),
                ("sgt", "sergeant"),
                ("capt", "captain"),
                ("esq", "esquire"),
                ("ltd", "limited"),
                ("col", "colonel"),
                ("ft", "fort"),
            ]
        ]

        text = re.sub(" +", " ", text)

        for regex, replacement in _abbreviations:
            text = re.sub(regex, replacement, text)
        return text
