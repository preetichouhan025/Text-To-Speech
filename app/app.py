from TTSInferencing import TTSInferencing
from speechbrain.inference.vocoders import HIFIGAN
# import torchaudio
import  streamlit as st
import numpy as np


tts_model = TTSInferencing.from_hparams(source="./",
                                    hparams_file='./hyperparams.yaml',
                                    pymodule_file='./module_classes.py',
                                    # savedir="./",
                                    )

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")

# text = ["Hello I am a girl", "How is your day going", "I hope you are doing well"]

# Input text
text_input = st.text_input("Enter your text here")

# Check if the input is a list
if isinstance(text_input, str):
    # Convert the input to a list
    text = [text_input]
else:
    text = text_input

if st.button("Synthesize Speech"):
    if text:
        mel_outputs = tts_model.encode_batch(text)
        waveforms = hifi_gan.decode_batch(mel_outputs)
    
        waveform =  waveforms[0].squeeze(1).numpy()
    
        # Normalize the waveform to the range [-1, 1] if necessary
        if np.max(np.abs(waveform)) > 1.0:
            waveform /= np.max(np.abs(waveform))
    
        # Display the audio widget to play the synthesized speech
        st.audio(waveform, format="audio/wav", sample_rate = 22050)
    else:
        st.error("Please enter text to get the speech.")