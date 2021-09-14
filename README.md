
# Wav2Vec2CTC With KenLM
Using KenLM ARPA language model with beam search to decode audio files and show the most probable transcription.

Assuming you've already installed [HuggingFace transformers library](https://github.com/huggingface/transformers), 
you need also to install the ctcdecode library:
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

Then, you need to change the language model path from inside the script `wav2vec2_kenlm.py`:
```
lm_path = "YOUR ARPA LANGUAGE MODEL PATH"
```
You may download a pretrained ARPA English Language model from [this link](https://kaldi-asr.org/models/m5).

Finally, run the script and see the result:
```
python wav2vec2_kenlm.py
```
