# Automatic Page Turner

Final project for CPSC459/559

## I. Dependencies

Run the following command in the terminal to get the dependencies:

    $ pip install -r requirements.txt

## II. Usage

Requires Python 3.7.2. 

Run the following command in the terminal to launch the web app

    $ export FLASK_APP=app.py
    $ flask run

Then you can load sheet music images!

Currently, we are restricting the scope: the system is only capable of recognizing and representing high resolution sheet music written for a single monophonic musical instrument using note or rest values equal to or greater than sixteenth notes, expressed on a staff consisting of either the treble or the bass clef and one of the common time signatures, i.e. common time, 2/4, 4/4, 3/4, 3/2 or 6/8. Consequently, my recognition system cannot perform *key* or *time signature alterations*, or *detect tempo demarcations*, *harmony*, *multi-instrumentation*, *braced-joined staffs*, *tuplets*, *repeats*, *slurs* and *ties*, *articulation* and *dynamic marks*, or *dotted rhythms*.

![alt text](https://github.com/anyati/cadenCV/blob/master/resources/README/image4.jpg)
Standard input to the recognition system.
