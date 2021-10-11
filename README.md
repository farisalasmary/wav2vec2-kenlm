
# Wav2Vec2CTC With KenLM
Using KenLM ARPA language model with beam search to decode audio files and show the most probable transcriptions.

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

To find words boundaries, you need to install the [CTC segmentation](https://github.com/lumaku/ctc-segmentation) library:
```
pip install ctc-segmentation
```

Finally, run the script and see the result:
```
python wav2vec2_kenlm.py
```

## Acknowledgments
 This project uses the functionalities of different open-source projects that are mentioned below.
* [CTC beam search decoder in C++ with PyTorch bindings](https://github.com/parlance/ctcdecode)
* [decoder.py file](https://github.com/SeanNaren/deepspeech.pytorch)
* [Another implementation of beam search decoder in pure Python](https://github.com/Wikidepia/wav2vec2-indonesian/blob/master/notebooks/kenlm-wav2vec2.ipynb)
* [CTC segmentation](https://github.com/lumaku/ctc-segmentation)