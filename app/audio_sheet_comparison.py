import numpy as np, scipy, matplotlib.pyplot as plt
import librosa
import librosa.display
import math
import pyaudio

from .audio_pro import note_to_midi, midi_to_2d, smooth_audio, merge_duplicate
# File comparison

def audio_sheet_compare(audio, sheet, co = 0.1):
    i = 0  # audio index
    j = 0  # sheet index
    ratio = None
    
    while i < audio.shape[0]:
        if audio[i, 0] != sheet[0, 0]: i +=1
        else:
            # find the first note match in audio and sheet
            # check if the following notes in the audio matches the sheet
            while j < sheet.shape[0] and i < audio.shape[0]:
                if audio[i, 0] == sheet[j, 0]:
                    if ratio == None: 
                        ratio = audio[i, 1]/sheet[j, 1]
                        i += 1
                        j += 1
                    elif np.abs((ratio-audio[i, 1]/sheet[j, 1])/ratio) < co:
                        if j == sheet.shape[0] -1: return True
                        else: 
                            i += 1
                            j += 1
                    else: 
                        i += 1
                        j = 0
                        break
                else:
                    i+=1
                    j = 0
                    break
    return False

def file_compare(file, sheet):
	# audio
	x, sr = librosa.load(file)
	bins_per_octave = 12
	cqt = librosa.cqt(x, sr=sr, n_bins=60, bins_per_octave=bins_per_octave)
	log_cqt = librosa.amplitude_to_db(np.abs(cqt))
	a = []
	for i in range(len(log_cqt.argmax(0))):
	    k = log_cqt.argmax(0)[i]
	    a.append(k+24)
	audio = audio_pro.smooth_audio(a)

	#sheet process
	midi = audio_pro.note_to_midi(sheet)
	sheet = audio_pro.midi_to_2d(midi)

	return audio_sheet_compare(audio, sheet)

def realtime_pitch(stream, buffer_size):
    audiobuffer = stream.read(buffer_size)
    signal = np.fromstring(audiobuffer, dtype=np.float32)
    bins_per_octave = 12
    cqt = librosa.cqt(signal, sr=sr, n_bins=60, bins_per_octave=bins_per_octave)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    return log_cqt.argmax(0)[1]


def stream_compare(sheet):
	# still in progress
	p = pyaudio.PyAudio()

	# open stream
	buffer_size = 1024
	pyaudio_format = pyaudio.paFloat32
	n_channels = 1
	sr = 44100
	record_duration = 5 # exit
	stream = p.open(format=pyaudio_format,
	                channels=n_channels,
	                rate=sr,
	                input=True,
	                frames_per_buffer=buffer_size)

	status = False

	print("*** starting recording")
	num_frames = 0
	index = 0
	ratio = None

	# while True: 
	#     print(realtime_pitch(stream, buffer_size))

	while index < sheet.shape[0]:
	    if index == 0:    
	        p = realtime_pitch(stream, buffer_size)
	        print(p)
	        num_frames +=1
	        while p != sheet[index, 0]: p = realtime_pitch(stream, buffer_size)
	        print("first note match!")
	    
	    audio_len = 0
	    while p == sheet[index, 0]:
	        audio_len += 1
	        p = realtime_pitch(stream, buffer_size)
	        num_frames += 1
	    
	    print("length of note = %d"%audio_len)
	        
	    if p == sheet[index+1, 0]: index += 1
	        
	    if ratio == None: ratio = sheet[0, 1]/audio_len
	    elif ratio == sheet[index, 1]/audio_len:
	        if index == sheet.shape[0] - 1: 
	            return True
	            break
	    else: index = 0
	        
	    if num_frames > sr * record_duration / 3: break

	print("*** done recording")
	stream.stop_stream()
	stream.close()
	p.terminate()