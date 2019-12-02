import numpy as np, scipy, matplotlib.pyplot as plt
import librosa
import librosa.display
import math
import pyaudio
import time # for debugging

from .audio_pro import note_to_midi, midi_to_2d, smooth_audio, merge_duplicate

buffer_size = 1024
pyaudio_format = pyaudio.paFloat32
n_channels = 1
sr = 44100

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
	cqt = librosa.cqt(x, sr=44100, n_bins=60, bins_per_octave=bins_per_octave)
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
    audiobuffer = stream.read(buffer_size, exception_on_overflow = False)
    signal = np.fromstring(audiobuffer, dtype=np.float32)
    bins_per_octave = 12
    cqt = librosa.cqt(signal, sr=44100, n_bins=60, bins_per_octave=bins_per_octave)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    return log_cqt.argmax(0)[1]

def stream_compare(sheet):
	# time.sleep(4)
	# return True

	# initialise pyaudio
	# sheet 
	midi = note_to_midi(sheet)
	sheet = midi_to_2d(midi)[3:]

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

	# print(sheet)

	print("*** starting recording")
	num_frames = 0
	index = 0
	ratio = None

	while True:
	    try:
	        s = realtime_pitch(stream, buffer_size)
	        while s!= sheet[0, 0]: s = realtime_pitch(stream, buffer_size)
	        
	        while index < sheet.shape[0]:
	            l = 0
	            while s == sheet[index, 0]: 
	                l+=1
	                s = realtime_pitch(stream, buffer_size)
	            
	            t = math.ceil(l/3)
	            c = 0
	            while c < t:
	                s = realtime_pitch(stream, buffer_size)
	                c += 1
	                if index + 1 < sheet.shape[0] and s == sheet[index+1, 0]: 
	                    break
	                if s == sheet[index, 0]:
	                    while s == sheet[index, 0]: 
	                        l+=1
	                        s = realtime_pitch(stream, buffer_size)
	                    c = 0
	            
	            if ratio == None: 
	                ratio = l/sheet[index, 1]
	                index += 1
	            elif np.abs((ratio-l/sheet[index, 1])/ratio) < 0.5:
	                if index == sheet.shape[0] -1: 
	                    return True
	                else: 
	                    index += 1
	            else: 
	            	if index == sheet.shape[0] -1 and l > 15: return True
	            	index = 0
	            	break
	    except KeyboardInterrupt:
	        print("*** Ctrl+C pressed, exiting")
	        break

	print("*** done recording")
	stream.stop_stream()
	stream.close()
	p.terminate()
	return False