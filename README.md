# Automatic Page Turner

Final project for CPSC459/559

## I. Dependencies

Run the following command in the terminal to get the dependencies:

    $ pip install -r requirements.txt
    
Note, to install PyAudio on MacOS with Python 3.7.2, do the following:

    $ pip install portaudio

    # create $HOME/.pydistutils.cfg using the include and lib directories of your portaudio install
    $ vim $HOME/.pydistutils.cfg

In the config file, write the following 3 lines:

[build_ext]
include_dirs=/usr/local/Cellar/portaudio/19.6.0/include/ (Note: this will vary according to your own system)
library_dirs=/usr/local/Cellar/portaudio/19.6.0/lib/ (Note: ditto)

    $ pip install --global-option='build_ext' --global-option='-I/usr/local/include' --global-option='-L/usr/local/lib' pyaudio

Then you'll be all set to use PyAudio!

## II. Usage

Requires Python 3.7.2. 

Run the following command in the terminal to launch the web app

    $ export FLASK_APP=app.py
    $ flask run

Then you can load sheet music images!

Currently, we are restricting the scope: the system is only capable of recognizing and representing high resolution sheet music written for a single monophonic musical instrument using note or rest values equal to or greater than sixteenth notes, expressed on a staff consisting of either the treble or the bass clef and one of the common time signatures, i.e. common time, 2/4, 4/4, 3/4, 3/2 or 6/8. Consequently, my recognition system cannot perform *key* or *time signature alterations*, or *detect tempo demarcations*, *harmony*, *multi-instrumentation*, *braced-joined staffs*, *tuplets*, *repeats*, *slurs* and *ties*, *articulation* and *dynamic marks*, or *dotted rhythms*.

![alt text](https://github.com/anyati/cadenCV/blob/master/resources/README/image4.jpg)
Standard input to the recognition system.
