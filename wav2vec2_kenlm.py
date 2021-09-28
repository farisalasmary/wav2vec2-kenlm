
"""
    @author
          ______         _                  _
         |  ____|       (_)           /\   | |
         | |__ __ _ _ __ _ ___       /  \  | | __ _ ___ _ __ ___   __ _ _ __ _   _
         |  __/ _` | '__| / __|     / /\ \ | |/ _` / __| '_ ` _ \ / _` | '__| | | |
         | | | (_| | |  | \__ \    / ____ \| | (_| \__ \ | | | | | (_| | |  | |_| |
         |_|  \__,_|_|  |_|___/   /_/    \_\_|\__,_|___/_| |_| |_|\__,_|_|   \__, |
                                                                              __/ |
                                                                             |___/
            Email: farisalasmary@gmail.com
            Date:  Sep 15, 2021
"""

"""
This code uses some of the works in the following repos:
https://github.com/parlance/ctcdecode
https://github.com/SeanNaren/deepspeech.pytorch
https://github.com/Wikidepia/wav2vec2-indonesian/blob/master/notebooks/kenlm-wav2vec2.ipynb
"""

from decoder import *
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import librosa


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Device: {device}')

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

print(f'Loading Wav2Vec2CTC Model: "{MODEL_ID}"')
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

vocab_dict = processor.tokenizer.get_vocab()
sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())

# Lower case ALL letters
vocab = []
for _, token in sort_vocab:
    vocab.append(token.lower())

# replace the word delimiter with a white space since the white space is used by the decoders
vocab[vocab.index(processor.tokenizer.word_delimiter_token)] = ' '

# you can download the following ARPA LM from this link:
# "https://kaldi-asr.org/models/5/4gram_big.arpa.gz"
lm_path = "data/local/local_lm/data/arpa/4gram_big.arpa" 

# alpha, beta, and beam_wdith SHOULD be tuned on the dev-set to get the best settings
# Feel free to check other inputs of the BeamCTCDecoder
alpha=0
beta=0

beam_width = 1024

beam_decoder = BeamCTCDecoder(vocab, lm_path=lm_path,
                                 alpha=alpha, beta=beta,
                                 cutoff_top_n=40, cutoff_prob=1.0,
                                 beam_width=beam_width, num_processes=16,
                                 blank_index=vocab.index(processor.tokenizer.pad_token))


greedy_decoder = GreedyDecoder(vocab, blank_index=vocab.index(processor.tokenizer.pad_token))


# load the test audio file
audiofile_path = 'english_sample.wav'

print(f'Load audio file: "{audiofile_path}"')
speech_array, sampling_rate = librosa.load(audiofile_path, sr=16_000)
inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)

net_input = inputs.to(device).input_values
net_input = net_input.to(device)

print("Model Prediction...")
model = model.eval().to(device)
with torch.no_grad():
    logits = model(net_input, attention_mask=inputs.attention_mask).logits


print('Decoding using Beam Search Decoder....')
beam_decoded_output, beam_decoded_offsets = beam_decoder.decode(logits)

print('Decoding using Greedy Decoder....')
greedy_decoded_output, greedy_decoded_offsets = greedy_decoder.decode(logits)


print('Greedy Decoding Output:', greedy_decoded_output[0][0])
print('#'*85)
print('Beam Search Decoding Output:', beam_decoded_output[0][0]) # print the top prediction of the beam search


