import torch
import gradio as gr
import json
import openai

from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils_vits
import commons
import sys
import re
from torch import no_grad, LongTensor
import logging
from winsound import PlaySound

from pygtrans import Translate, Null
import time


############################################################################
def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)

def print_speakers(speakers, escape=False):
    if len(speakers) > 100:
        return
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name, escape)

def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id

def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text

def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text

def generateSound(inputString):
    idmessage = """ID      Speaker
0       maho
"""
    speakerID = 0
    model = r".\vits_models\G.pth"
    config = r".\vits_models\config.json"        
    if "image" and "png" in inputString:
        return False
    hps_ms = utils_vits.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils_vits.load_checkpoint(model, net_g_ms)

    def voice_conversion():
        audio_path = input('Path of an audio file to convert:\n')
        print_speakers(speakers)
        audio = utils_vits.load_audio_to_torch(
            audio_path, hps_ms.data.sampling_rate)

        originnal_id = get_speaker_id('Original speaker ID: ')
        target_id = get_speaker_id('Target speaker ID: ')
        out_path = input('Path to save: ')

        y = audio.unsqueeze(0)

        spec = spectrogram_torch(y, hps_ms.data.filter_length,
                                 hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
                                 center=False)
        spec_lengths = LongTensor([spec.size(-1)])
        sid_src = LongTensor([originnal_id])

        with no_grad():
            sid_tgt = LongTensor([target_id])
            audio = net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[
                0][0, 0].data.cpu().float().numpy()
        return audio, out_path

    if n_symbols != 0:
        if not emotion_embedding:
            #while True:
            if(1==1):
                #choice = input('TTS or VC? (t/v):')
                choice = 't'
                if choice == 't':
                    #text = input('Text to read: ')
                    text = inputString
                    client = Translate()
                    while True:
                        text_t = client.translate(text, target='ja')
                        if isinstance(text, Null):
                            print("Translation failure!")
                            time.sleep(2)
                        else:
                            print("Translation Success!")
                            text = text_t.translatedText
                            break
                    if text == '[ADVANCED]':
                        #text = input('Raw text:')
                        text = "I can't speak!"
                        #print('Cleaned text is:')
                        #ex_print(_clean_text(
                        #    text, hps_ms.data.text_cleaners), escape)
                        #continue

                    length_scale, text = get_label_value(
                        text, 'LENGTH', 1.1, 'length scale')
                    noise_scale, text = get_label_value(
                        text, 'NOISE', 0.2, 'noise scale')
                    noise_scale_w, text = get_label_value(
                        text, 'NOISEW', 0.8, 'deviation of noise')
                    cleaned, text = get_label(text, 'CLEANED')

                    stn_tst = get_text(text, hps_ms, cleaned=cleaned)

                    #print_speakers(speakers, escape)
                    #speaker_id = get_speaker_id('Speaker ID: ')
                    speaker_id = speakerID 
                    #out_path = input('Path to save: ')
                    out_path = "output.wav"

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

                elif choice == 'v':
                    audio, out_path = voice_conversion()

                write(out_path, hps_ms.data.sampling_rate, audio)
                print('Successfully saved!')
                return True
