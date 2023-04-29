from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os
import numpy as np
from threading import Thread
import matplotlib.pyplot as plt
import takeAtt



global capture,rec_frame, grey, switch, neg, face, rec, out,attended,identefier,this_frame
this_frame=0
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
identefier = '0'
attended=False
global first_Capture
first_Capture = False
#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/face_detect/deploy.prototxt.txt', './saved_model/face_detect/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)








def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)




def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
    

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    global this_frame
    while True:
        success, frame = camera.read() 
        this_frame = frame
        if success:
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

               
        else:
            pass
        



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture,this_frame,attended,first_Capture,identefier
            identefier = request.form['identefier']
            this_frame = detect_face(this_frame)
            now = datetime.datetime.now()
            p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
            cv2.imwrite(p, this_frame)
            attended = takeAtt.take_attendance(identefier, p)
            os.remove(p)
            first_Capture = True
            
            
    elif request.method=='GET':
        return render_template('index.html')
    if(first_Capture & attended):
        camera.release()
        cv2.destroyAllWindows()
        return render_template("lecture.html",id = identefier)
    else:
        return render_template('index.html',wrong = "wrong input try again")



if __name__ == '__main__':
    app.run()