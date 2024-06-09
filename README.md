# Text-To-Speech (Neural Speech Synthesis with Transformer Network)

This repository provides the necessary tools for Text-to-Speech (TTS) with SpeechBrain using a Transformer pretrained on LJSpeech

## Introduction

Text-to-speech (TTS) is the task of creating natural-like speech or audio from entered text, where speech can be generated for multiple speakers and in multiple languages. There have been many models based on conventional ways like RNNs but the recent sequence-to-sequence architecture of Transformers has made the task much more efficient.

The main goal of this project is to use the transformers for the TTS task. And use the multi-headed self-attention mechanism to capture the long-term dependent features and also, improve the training with parallelization. This specific implementation is similar to Tacotron2 but it replaces the RNN structures and the attention mechanism used. Rather in the project, phonemes were used as an input to the transformer-based architecture that generates mel spectrograms. The output audios are generated after the mel spectrograms are fed to the WaveNet vocoder. Experiments were conducted on the LJSpecch dataset to test the implemented network. Since both the encoder and decoder are constructed in parallel, it improved the training efficiency as compared to Tacotron2.

In this implementation, we have two main components: an encoder and a decoder. The input text is first converted into a sequence of phonemes and then the positional encoding is applied before passing it to the transformer encoder. The hidden states encoder output along with the mel spectrogram of audios are used as part of the teacher forcing method to train the decoder. The transformer decoders are non-autoregressive at training and autoregressive at inferencing. With the multi-headed transformer architecture, the self-attention mechanism integrates the sequential dependency on the last previous hidden state to improve the parallelization process to improve the efficiency and speed and handle the long-term dependency issue as well. With different heads, the context vector is built from different aspects, hence providing better output.

## Problem Statment

Natural-sounding speech generation is a difficult task to achieve due to many reasons including - multiple languages and their native pronunciations and expressiveness. The intonation, rhythm, coherence, variability, and diversity; all play a crucial role in speech synthesis. The computational complexity and resources required especially for lengthy or complicated sentences are too high.

## Transformer Architecture

<p>
    <img src="/images/transformer.JPG" width="300" height="600" />
</p>

## Dataset

The LJSpeech dataset (2.6 GB) consists of 13,100 short audio clips of a single speaker. A transcription is provided for each audio clip.

## Implemented System Architecture - Neural Speech Synthesis with Transformer Network

<p>
    <img src="/images/TTS.JPG" width="300" height="600" />
</p>

## Modules Implemented

Transformers have outperformed the RNN-based models in natural language processing tasks significantly. The goal of this project is to implement a transformer for natural speech synthesis (figure 2). Compared to RNN-based architecture, the transformer has two main advantages: 1) parallel training and 2) self-attention to capture the long-term context of input sequences.

The complete text-to-speech system basically converts the given input text into phonemes and then the encoder maps these tokens into latent representations to generate a sequence of hidden states from the encoder. The decoder then takes these sequences of hidden units along with mel spectrograms (teacher forcing)  to output the mel spectrogram as part of training. The decoder is autoregressive, therefore a sequential greedy approach is used here.

<u>**Following are the modules implemented as part of this project:**</u>


1.   ***Text-to-Phoneme Conversion:***  Phonemes are very useful since they are natural speech targets as they represent basic sounds. So the tokenization is done via phonemes using the pre-trained model - GraphemeToPhoneme. The training data is enough to learn each phoneme and generalize well to new words.

2.   ***Scaled Positional Encoding:***   Here, triangle positional embeddings with trainable weights are implemented so that the embeddings can adaptively fit the scales of both encoder and decoder prenet outputs.

3.   ***Encoder Prenet:***    It is a three-layer CNN, followed by a batch normalization layer and dropout layer applied to input embeddings. It captures the long-term context in the input. The module_class named CNN Encoder Prenet implements this logic. The output of each convolutional layer is 512 dims. The activation used is ReLU.

4.   ***Decoder Prenet:***    This is used to process the mel spectrogram. It's 2 fully connected layered with each having 256 dims followed by a ReLU activation. The module class Decoder Prenet implements the same. The phonemes have trainable embeddings which are adaptive but the mel spectrogram is fixed. Therefore, decoder Prenet helps to project both the phonemes embeddings and mel spectrogram into the same subspace. Using them as a pair allows the attention mechanism of the transformer to work properly.

5.   ***Encoder:***   The non-autoregressive transformer Encoder is used which have multi-headed attention which allow parallel computing. It directly builds long-term dependency between two frames. Since each of them considers the global context of the complete sentence, audio prosody synthesis becomes better, especially for long sentences.

6.   ***Decoder:***    Here, a transformer decoder with multi-headed self-attention is used.
     
7.   ***Mel Linears, Stop Linears & Postnet***   Finally, After getting the output from the decoder, there are two linear operations to be applied, The first is for the stop_token predictions so a stop_linear is used with n_neurons = 1 and a mel_linear with n_neruons = 80 is used for the mel spectrogram. After getting the output from the mel_linear, it is passed through a Decoder Postnet which is a 5-layer CNN and is used to refine the construction of the mel spectrogram.

8.    ***Loss:***    Mean Squared Error is used for mel_spectogram and Binary cross-entropy is used for stop token prediction. The final loss is the sum of both of the losses.

9.   ***Wavenet Vocoder***     Wavnet Vocoder is a pre-trained model that is used to transform a waveform into an audio. The vocoder used is - HIFIGAN

<u> Following table shows the experimental setup details for each training round: </u>




**1.  Data size for each training round**


| Training Sess | Training Set     | Validation Set |   Train Set  |
|---------------|------------------|----------------|--------------|
| Training 1    |      4000        |      700       |   500        |
| Training 2    |      10,462      |      1282      |   1356       |
| Training 3    |      10,462      |      1282      |   1356       |



**2.  Hyperparameters**


| Training Sess | Number of Epochs | Batch Size |Learning Rate | Optimizer |
|---------------|------------------|------------|--------------|-----------|
| Training 1    |        30        |      16    |   0.001      |   SGD    |
| Training 2    |        30        |      08    |   0.0001     |   ADAM    |
| Training 3    |        80        |      08    |   0.00001    |   ADAM    |


## Results

<p>
    <img src="/images/Capture.JPG" width="700" height="300" />
</p>


## Challenges & Conclusion

Despite the improvements and advancements in TTS tasks because of Transformers-based architecture, many challenges still persist. The challenges in general include speech generation in multiple languages with their native accents, synthesis latency for real-time applications, and the computational requirements of large models with millions of parameters.

Considering the limited computation, the model wasn't trained on enough encoders and decoders and with fewer epochs. And it is due to this reason that even though the loss is less, the speech generated isn't the targeted output. If the system was trained on more parameters then the overall loss could have been improved along with the performance of the model.

The next step in future improvements of this task should be concentrating on enhancing the speaker adaption capabilities, developing lightweight models for deployment on small devices like mobile phones, and developing the computational devices required for the aforementioned large models.


## **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

## **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
