import numpy as np, scipy, matplotlib.pyplot as plt
import librosa
import librosa.display
import math
import pyaudio

def note_to_midi(x):
    midi = []
    for i in x:
        if i == 'A0': midi.append(21)
        if i == 'A#0': midi.append(22)
        if i == 'B0': midi.append(23)
        if i == 'C1': midi.append(24)
        if i == 'C#1': midi.append(25)
        if i == 'D1': midi.append(26)
        if i == 'D#1': midi.append(27)
        if i == 'E1': midi.append(28)
        if i == 'F1': midi.append(29)
        if i == 'F#2': midi.append(42)
        if i == 'F1': midi.append(29)
        if i == 'F#1': midi.append(30)
        if i == 'E1': midi.append(28)
        if i == 'F1': midi.append(29)
        if i == 'F#1': midi.append(30)
        if i == 'G1': midi.append(31)
        if i == 'G#1': midi.append(32)
        if i == 'A1': midi.append(33)
        if i == 'A#1': midi.append(34)
        if i == 'B1': midi.append(35)
        if i == 'C2': midi.append(36)
        if i == 'C#2': midi.append(37)
        if i == 'D2': midi.append(38)
        if i == 'D#2': midi.append(39)
        if i == 'E2': midi.append(40)
        if i == 'F2': midi.append(41)
        if i == 'F#2': midi.append(42)
        if i == 'G2': midi.append(43)
        if i == 'G#2': midi.append(44)
        if i == 'A2': midi.append(45)
        if i == 'A#2': midi.append(46)
        if i == 'B2': midi.append(47)
        if i == 'C3': midi.append(48)
        if i == 'C#3': midi.append(49)
        if i == 'D3': midi.append(50)
        if i == 'D#3': midi.append(51)
        if i == 'E3': midi.append(52)
        if i == 'F3': midi.append(53)
        if i == 'F#3': midi.append(54)
        if i == 'G3': midi.append(55)
        if i == 'G#3': midi.append(56)
        if i == 'A3': midi.append(57)
        if i == 'A#3': midi.append(58)
        if i == 'B3': midi.append(59)
        if i == 'C4': midi.append(60)
        if i == 'C#4': midi.append(61)
        if i == 'D4': midi.append(62)
        if i == 'D#4': midi.append(63)
        if i == 'E4': midi.append(64)
        if i == 'F4': midi.append(65)
        if i == 'F#4': midi.append(66)
        if i == 'G4': midi.append(67)
        if i == 'G#4': midi.append(68)
        if i == 'A4': midi.append(69)
        if i == 'A#4': midi.append(70)
        if i == 'B4': midi.append(71)            
        if i == 'C5': midi.append(72)
        if i == 'C#5': midi.append(73)
        if i == 'D5': midi.append(74)
        if i == 'D#5': midi.append(75)
        if i == 'E5': midi.append(76)
        if i == 'F5': midi.append(77)
        if i == 'F#5': midi.append(78)
        if i == 'G5': midi.append(79)
        if i == 'G#5': midi.append(80)
        if i == 'A5': midi.append(81)
        if i == 'A#5': midi.append(82)
        if i == 'B5': midi.append(83)
        if i == 'C6': midi.append(84)
        if i == 'C#6': midi.append(85)
        if i == 'D6': midi.append(86)
        if i == 'D#6': midi.append(87)
        if i == 'E6': midi.append(88)
        if i == 'F6': midi.append(89)
        if i == 'F#6': midi.append(90)
        if i == 'G6': midi.append(91)
        if i == 'G#6': midi.append(92)
        if i == 'A6': midi.append(93)
        if i == 'A#6': midi.append(94)
        if i == 'B6': midi.append(95)
        if i == 'C7': midi.append(96)
        if i == 'C#7': midi.append(97)
        if i == 'D7': midi.append(98)
        if i == 'E7': midi.append(100)
        if i == 'F7': midi.append(101)
        if i == 'F#7': midi.append(102)
        if i == 'G7': midi.append(103)
        if i == 'G#7': midi.append(104)
        if i == 'A7': midi.append(105)
        if i == 'A#7': midi.append(106)
        if i == 'B7': midi.append(107)
        if i == 'C8': midi.append(108)
    return midi

def midi_to_2d(midi):
    d = []
    i = 0
    while i < len(midi) - 1:
        p = midi[i]
        l = 1
        while midi[i+1] == midi[i]: 
            i += 1
            l += 1
            if i+1 == len(midi): break
        i += 1
        d.append([p, l])
    return np.matrix(d)

def smooth_audio(audio):
    delete_list = []
    for i in range(audio.shape[0]):
        if audio[i, 1] <= 6:
            if i+1 < audio.shape[0] and np.abs(audio[i+1, 0] - audio[i, 0])<=1: audio[i+1, 1] += 1
            delete_list.append(i)
    audio = np.delete(audio, delete_list, 0)
    return merge_duplicate(audio)

def merge_duplicate(audio):
    # merge those notes with same pitches
    delete_list = []
    for i in range(audio.shape[0]-1):
        if audio[i, 0] == audio[i+1, 0]:
            audio[i+1, 1] += audio[i, 1]
            delete_list.append(i)
    return np.delete(audio, delete_list, 0)

