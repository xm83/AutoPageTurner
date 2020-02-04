This folder contains the first version of the app, initially built this app as the final project for CPSC459/559 "Building Interactive Machines", where we combined computer vision algorithms (mainly template matching) with audio processing techniques to create an integrated system. We won the [Class Choice Award](https://cpsc459-bim.gitlab.io/f19/projects_list/) :trophy:, given to the top project in class. For a demo, check out this [video](https://youtu.be/lQWgagWgHMw) :clapper:. For more technical details, check out our accompanying [paper](https://cpsc459-bim.gitlab.io/f19/assets/reports/music.pdf) :page_with_curl:.

## :musical_note: Dependencies 

Run the following command in the terminal to get the dependencies:

    $ pip install -r requirements.txt
    
Note, to install PyAudio on MacOS with Python 3.7.2, run the following commands:

    $ pip install portaudio

    # create $HOME/.pydistutils.cfg using the include and lib directories of your portaudio install
    $ touch $HOME/.pydistutils.cfg

In the config file, write the following 3 lines:

[build_ext]

include_dirs=/usr/local/Cellar/portaudio/19.6.0/include/ (Note: this will vary according to your own system)

library_dirs=/usr/local/Cellar/portaudio/19.6.0/lib/ (Note: ditto)

Then run:

    $ pip install --global-option='build_ext' --global-option='-I/usr/local/include' --global-option='-L/usr/local/lib' pyaudio

Then you'll be all set to use PyAudio!

(Another easy way to manage all the requirements is to use an Anaconda virtual environment:
conda create -n [name of virtual env]
source activate [name of virtual env]
pip install -r requirements.txt
...etc.
)

## :musical_note:  Usage 

Requires Python 3.7.2. 

When you're in the app/ directory, run the following command in the terminal to launch the web app:

    $ export FLASK_APP=app.py
    $ export FLASK_DEBUG=1 # do this if you want to turn on debugging mode for development purposes
    $ flask run

Then you can load sheet music images!

Currently, we are restricting the scope: the system is only capable of recognizing and representing high resolution sheet music written for a single monophonic musical instrument using note or rest values equal to or greater than sixteenth notes, expressed on a staff consisting of either the treble or the bass clef and one of the common time signatures, i.e. common time, 2/4, 4/4, 3/4, 3/2, 6/4, or 6/8. Consequently, the recognition system cannot perform *key* or *time signature alterations*, or *detect tempo demarcations*, *harmony*, *multi-instrumentation*, *braced-joined staffs*, *tuplets*, *repeats*, *slurs* and *ties*, *articulation* and *dynamic marks*, or *dotted rhythms*.

![alt text](https://github.com/anyati/cadenCV/blob/master/resources/README/image4.jpg)
Standard input to the recognition system.
