
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
            Date:  Oct 11, 2021
"""

import torch
import librosa
import ctc_segmentation


def load_audio_files(audio_files_paths):
    batch_audio_files = []
    for audio_file_path in audio_files_paths:
        speech_array, sampling_rate = librosa.load(audio_file_path, sr=16_000)
        batch_audio_files.append(speech_array)
    
    return batch_audio_files, sampling_rate


def get_logits(batch_audio_files, model, processor, device='cpu'):
    processed_batch = processor(batch_audio_files, sampling_rate=16_000, return_tensors="pt", padding=True).input_values
    net_input = processed_batch.to(device)
    net_input = net_input.to(device)
    
    model = model.eval().to(device)
    with torch.no_grad():
        logits = model(net_input).logits
    
    # the Wav2Vec2Processor will pad the batch with the max signal length in the batch
    # so that ALL audio files have the same length
    max_signal_length = processed_batch.shape[1]
    
    return logits, max_signal_length



def get_segments(logits, decoded_output, max_signal_length, sampling_rate, vocab_list):
    
    # CTC log posteriors inference
    with torch.no_grad():
        softmax = torch.nn.LogSoftmax(dim=-1)
        lpzs = softmax(logits).cpu().numpy()
    
    batch_segments_list = []
    for i in range(len(decoded_output)):
        lpz = lpzs[i] # select the logits of ith file
        transcription = decoded_output[i][0] # 0 means the most probable transcription
        text = transcription.split()
        
        # CTC segmentation preparation
        config = ctc_segmentation.CtcSegmentationParameters(char_list=vocab_list)
        config.index_duration = max_signal_length / lpz.shape[0] / sampling_rate
        
        # CTC segmentation
        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, text)
        timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, lpz, ground_truth_mat)
        segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)
        
        segments_list = []
        for word, segment in zip(text, segments):
            start_time, end_time, min_avg = segment
            segment_dict = {
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': (end_time - start_time),
                                'word': word
                          }
            
            segments_list.append(segment_dict)
        
        batch_segments_list.append(segments_list)
    
    return batch_segments_list

