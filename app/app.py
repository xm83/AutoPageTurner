from flask import Flask, Response,render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import os
import pyaudio
import numpy
import cv2
import matplotlib

from parse_img import parse

app = Flask(__name__)
# set the backend to a non-interactive one so that your server does not try to create (and then destroy) GUI windows
matplotlib.pyplot.switch_backend('Agg')  

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024 # how many samples in a frame that stream will read
RECORD_SECONDS = 5


audio1 = pyaudio.PyAudio()

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
app.config['SECRET_KEY'] = 'lolthisisasupersecretkeyhehehehe'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

dropzone = Dropzone(app)
# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# def genHeader(sampleRate, bitsPerSample, channels):
#     datasize = 2000*10**6
#     o = bytes("RIFF",'ascii')                                               # (4byte) Marks file as RIFF
#     o += (datasize + 36).to_bytes(4,'little')                               # (4byte) File size in bytes excluding this and RIFF marker
#     o += bytes("WAVE",'ascii')                                              # (4byte) File type
#     o += bytes("fmt ",'ascii')                                              # (4byte) Format Chunk Marker
#     o += (16).to_bytes(4,'little')                                          # (4byte) Length of above format data
#     o += (1).to_bytes(2,'little')                                           # (2byte) Format type (1 - PCM)
#     o += (channels).to_bytes(2,'little')                                    # (2byte)
#     o += (sampleRate).to_bytes(4,'little')                                  # (4byte)
#     o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4,'little')  # (4byte)
#     o += (channels * bitsPerSample // 8).to_bytes(2,'little')               # (2byte)
#     o += (bitsPerSample).to_bytes(2,'little')                               # (2byte)
#     o += bytes("data",'ascii')                                              # (4byte) Data Chunk Marker
#     o += (datasize).to_bytes(4,'little')                                    # (4byte) Data size in bytes
#     return o

# @app.route('/audio')
# def audio():
    # start recording
    # def sound():

    #     CHUNK = 1024
    #     sampleRate = 44100
    #     bitsPerSample = 16
    #     channels = 2
    #     wav_header = genHeader(sampleRate, bitsPerSample, channels)

    #     stream = audio1.open(format=FORMAT, channels=CHANNELS,
    #                     rate=RATE, input=True,input_device_index=1,
    #                     frames_per_buffer=CHUNK)
    #     print("recording...")
    #     #frames = []

    #     while True:
    #         data = wav_header+stream.read(CHUNK)
    #         yield(data)

    # return Response(sound())


@app.route('/', methods=['GET', 'POST'])
def index():
    
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']
    # handle image upload from Dropzone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            
            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename    
            )
            file.seek(0)
            b_str = file.read()
            if len(b_str) > 0:
                # read in the uploaded image
                img = cv2.imdecode(numpy.fromstring(b_str, numpy.uint8), 0)
                # parse the image to get the pitch duration array
                result = parse(img)
                # print("result from parse(img): ", result)
            else:
                print("ERROR: reading an empty byte string from img file with name: ", file.filename)

            # append image urls
            file_urls.append(photos.url(filename))

            
        session['file_urls'] = file_urls
        return "uploading..."
    # return dropzone template on GET request

    # results = []
    # for img in file_urls:
    #     img = cv2.imdecode(numpy.fromstring(request.files['file'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
    #     results.append(parse(img))
    # print(results)

    return render_template('index.html')


@app.route('/results')
def results():
    
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
        
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    session.pop('file_urls', None)

    # print(file_urls)
    # ['http://127.0.0.1:5000/_uploads/photos/mary_3.jpg', 'http://127.0.0.1:5000/_uploads/photos/mhush_3.jpg']
    
    return render_template('results.html', file_urls=file_urls)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True,port=5000)


