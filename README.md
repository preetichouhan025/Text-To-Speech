# Text-To-Speech

## Introduction

Text-to-speech (TTS) is the task of creating natural-like speech or audio from entered text, where speech can be generated for multiple speakers and in multiple languages. There have been many models based on conventional ways like RNNs but the recent sequence-to-sequence architecture of Transformers has made the task much more efficient.

The main goal of this project is to use the transformers for the TTS task. And use the multi-headed self-attention mechanism to capture the long-term dependent features and also, improve the training with parallelization. This specific implementation is similar to Tacotron2 but it replaces the RNN structures and the attention mechanism used. Rather in the project, phonemes were used as an input to the transformer-based architecture that generates mel spectrograms. The output audios are generated after the mel spectrograms are fed to the WaveNet vocoder. Experiments were conducted on the LJSpecch dataset to test the implemented network. Since both the encoder and decoder are constructed in parallel, it improved the training efficiency as compared to Tacotron2.

In this implementation we have two main components: an encoder and a decoder. The input text is first converted into a sequence of phonemes and then the positional encoding is applied before passing it to the transformer encoder. The hidden states encoder output along with the mel spectrogram of audios are used as part of the teacher forcing method to train the decoder. The transformer decoders are non-autoregressive at training and autoregressive at inferencing. With the multi-headed transformer architecture, the self-attention mechanism integrates the sequential dependency on the last previous hidden state to improve the parallelization process to improve the efficiency and speed and handle the long-term dependency issue as well. With different heads, the context vector is built from different aspects, hence providing better output.

## Problem Statment

Natural-sounding speech generation is a difficult task to achieve due to many reasons including - multiple languages and their native pronunciations and expressiveness. The intonation, rhythm, coherence, variability, and diversity; all play a crucial role in speech synthesis. The computational complexity and resources required especially for lengthy or complicated sentences are too high.

## Dataset

The LJSpeech dataset (2.6 GB) consists of 13,100 short audio clips of a single speaker. A transcription is provided for each audio clip.
