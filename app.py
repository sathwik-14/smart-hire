# Import all the necessary libraries
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import re
import time
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from flask import Flask , render_template , request , url_for , jsonify , Response
from flask import Flask , render_template , request , url_for , jsonify , Response
from werkzeug.utils import redirect, secure_filename
from flask_mail import Mail , Message
from flask_mysqldb import MySQL
from pyresparser import ResumeParser
from fer import Video
from fer import FER
from video_analysis import extract_text , analyze_tone
from decouple import config
import nltk
import dlib
import os
import math


shape_predictor_path = './static/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor_path)



# Access the environment variables stored in .env file
MYSQL_USER = config('mysql_user')
MYSQL_PASSWORD = config('mysql_password')

# To send mail (By interviewee)
MAIL_USERNAME = config('mail_username')
MAIL_PWD = config('mail_pwd')

# For logging into the interview portal
COMPANY_MAIL = config('company_mail')
COMPANY_PSWD = config('company_pswd')

# Create a Flask app
app = Flask(__name__)



# App configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = MYSQL_USER
app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
app.config['MYSQL_DB'] = 'smarthire' 
user_db = MySQL(app)

mail = Mail(app)              
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = MAIL_USERNAME
app.config['MAIL_PASSWORD'] = MAIL_PWD
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_ASCII_ATTACHMENTS'] = True
mail = Mail(app)


# Initial sliding page
@app.route('/')
def home():
    return render_template('index.html')


# Interviewee signup 
@app.route('/signup' , methods=['POST' , 'GET'])
def interviewee():
    if request.method == 'POST' and 'usermail' in request.form and 'userpassword' in request.form:
        # username = request.form['username']
        usermail = request.form['usermail']
        userpassword = request.form['userpassword']

        cursor = user_db.connection.cursor()

        cursor.execute("SELECT * FROM candidates WHERE password = % s AND email = %s", (userpassword, usermail))
        account = cursor.fetchone()

        if account:
            reg = "You have successfully Logged In !!"
            return render_template('FirstPage.html' , reg = reg)
        # elif not re.fullmatch(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', usermail):
        #     err = "Invalid Email Address !!"
        #     return render_template('index.html' , err = err)
        # elif not re.fullmatch(r'[A-Za-z0-9\s]+', username):
        #     err = "Username must contain only characters and numbers !!"
        #     return render_template('index.html' , err = err)
        # elif not username or not userpassword or not usermail:
        #     err = "Please fill out all the fields"
        #     return render_template('index.html' , err = err)
        else:
            # cursor.execute("INSERT INTO candidates VALUES (NULL, % s, % s, % s)" , (username, usermail, userpassword,))
            # user_db.connection.commit()
            err = "Invalid Credentials"
            return render_template('index.html' , err = err)
    else:
        return render_template('index.html')


# Interviewer signin 
@app.route('/signin' , methods=['POST' , 'GET'])
def interviewer():
    if request.method == 'POST' and 'company_mail' in request.form and 'password' in request.form:
        
        company_mail = request.form['company_mail']
        password = request.form['password']

        if company_mail == COMPANY_MAIL and password == COMPANY_PSWD:
            with open('./static/result.json' , 'r') as file:
                output = json.load(file)
            with open('./static/results.json' , 'r') as file1:
                output1 = json.load(file1)
            return render_template('candidateSelect.html', output = output,output1=output1)
        else:
            return render_template("index.html" , err = "Incorrect Credentials")
    else:
        return render_template("index.html")


# personality trait prediction using Logistic Regression and parsing resume
@app.route('/prediction' , methods = ['GET' , 'POST'])
def predict():
    # get form data
    if request.method == 'POST':
        fname = request.form['firstname'].capitalize()
        lname = request.form['lastname'].capitalize()
        age = int(request.form['age'])
        gender = request.form['gender']
        email = request.form['email']
        file = request.files['resume']
        path = './static/{}'.format(file.filename)
        file.save(path)
        val1 = int(request.form['openness'])
        val2 = int(request.form['neuroticism'])
        val3 = int(request.form['conscientiousness'])
        val4 = int(request.form['agreeableness'])
        val5 = int(request.form['extraversion'])
        
        # model prediction
        df = pd.read_csv(r'static\trainDataset.csv')
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        x_train = df.iloc[:, :-1].to_numpy()
        y_train = df.iloc[:, -1].to_numpy(dtype = str)
        lreg = LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
        lreg.fit(x_train, y_train)

        if gender == 'male':
            gender = 1
        elif gender == 'female': 
            gender = 0
        input_df =  [gender, age, val1, val2, val3, val4, val5]
        
        pred = str(lreg.predict([input_df])[0]).capitalize()
        
        # get data from the resume
        data = ResumeParser(path).get_extracted_data()
        
        result = {'Name':fname+' '+lname , 'Age':age , 'Email':email , 'Mobile Number':data.get('mobile_number', None) , 
        'Skills':str(data['skills']).replace("[" , "").replace("]" , "").replace("'" , "") , 'Degree':data.get('degree' , None) , 'Designation':data.get('designation', None) ,
        'Total Experience':data.get('total_experience',None) , 'Predicted Personality':pred}

        with open('./static/result.json' , 'w') as file:
            json.dump(result , file)

    return render_template('questionPage.html')


# Record candidate's interview for face emotion and tone analysis
@app.route('/analysis', methods = ['POST'])
def video_analysis():

    # get videos using media recorder js and save
    quest1 = request.files['question1']
    quest2 = request.files['question2']
    quest3 = request.files['question3']
    path1 = "./static/{}.{}".format("question1","webm")
    path2 = "./static/{}.{}".format("question2","webm")
    path3 = "./static/{}.{}".format("question3","webm")
    quest1.save(path1)
    quest2.save(path2)
    quest3.save(path3)

    # speech to text response for each question - AWS
    responses = {'Question 1: Tell something about yourself': [] , 'Question 2: Why should we hire you?': [] , 'Question 3: Where Do You See Yourself Five Years From Now?': []}
    ques = list(responses.keys())

    text1 , data1 = extract_text("question1.webm")
    # extext_video("question1.webm")
    responses[ques[0]].append(text1)

    text2 , data2 = extract_text("question2.webm")
    responses[ques[1]].append(text2)

    text3 , data3 = extract_text("question3.webm")
    responses[ques[2]].append(text3)

    # tone analysis for each textual answer - IBM
    res1 = analyze_tone(text1)
    tones_doc1 = []

    for tone in res1['classifications']:
     
        tones_doc1.append((tone['class_name'] , round(tone['confidence']*100, 2)))
        # confidence=round(tone['confidence']*100, 2)
        # Results_data[email]["tones_1"]+={
        #      tone['class_name']:confidence
        # }

    if 'polite' not in [key for key, val in tones_doc1]:
        tones_doc1.append(('polite', 0.0))
    if 'satisfied' not in [key for key, val in tones_doc1]:
        tones_doc1.append(('satisfied', 0.0))
    if 'excited' not in [key for key, val in tones_doc1]:
        tones_doc1.append(('excited', 0.0))
    if 'sad' not in [key for key, val in tones_doc1]:
        tones_doc1.append(('sad', 0.0))
    if 'frustrated' not in [key for key, val in tones_doc1]:
        tones_doc1.append(('frustrated', 0.0))
    if 'sympathetic' not in [key for key, val in tones_doc1]:
        tones_doc1.append(('sympathetic', 0.0))
    if 'impolite' not in [key for key, val in tones_doc1]:
        tones_doc1.append(('frustrated', 0.0))
    tones_doc1 = sorted(tones_doc1)

    res2 = analyze_tone(text2)
    tones_doc2 = []

    for tone in res2['classifications']:
        tones_doc2.append((tone['class_name'] , round(tone['confidence']*100, 2)))
        # confidence1=round(tone['confidence']*100, 2)
        # Results_data[email]["tones_2"]+={
        #      tone['class_name']:confidence1
        # }
        
    if 'polite' not in [key for key, val in tones_doc2]:
        tones_doc2.append(('polite', 0.0))
    if 'satisfied' not in [key for key, val in tones_doc2]:
        tones_doc2.append(('satisfied', 0.0))
    if 'excited' not in [key for key, val in tones_doc2]:
        tones_doc2.append(('excited', 0.0))
    if 'sad' not in [key for key, val in tones_doc2]:
        tones_doc2.append(('sad', 0.0))
    if 'frustrated' not in [key for key, val in tones_doc2]:
        tones_doc2.append(('frustrated', 0.0))
    if 'sympathetic' not in [key for key, val in tones_doc2]:
        tones_doc1.append(('sympathetic', 0.0))
    if 'impolite' not in [key for key, val in tones_doc2]:
        tones_doc1.append(('impolite', 0.0))
    tones_doc2 = sorted(tones_doc2)

    res3 = analyze_tone(text3)
    tones_doc3 = []

    for tone in res3['classifications']:
        tones_doc3.append((tone['class_name'] , round(tone['confidence']*100, 2)))
        # confidence2=round(tone['confidence']*100, 2)
        # Results_data[email]["tones_1"]+={
        #      tone['class_name']:confidence2
        # }
        
    if 'polite' not in [key for key, val in tones_doc3]:
        tones_doc3.append(('polite', 0.0))
    if 'satisfied' not in [key for key, val in tones_doc3]:
        tones_doc3.append(('satisfied', 0.0))
    if 'excited' not in [key for key, val in tones_doc3]:
        tones_doc3.append(('excited', 0.0))
    if 'sad' not in [key for key, val in tones_doc3]:
        tones_doc3.append(('sad', 0.0))
    if 'frustrated' not in [key for key, val in tones_doc3]:
        tones_doc3.append(('frustrated', 0.0))
    if 'sympathetic' not in [key for key, val in tones_doc3]:
        tones_doc1.append(('sympathetic', 0.0))
    if 'impolite' not in [key for key, val in tones_doc3]:
        tones_doc1.append(('impolite', 0.0))
    tones_doc3 = sorted(tones_doc3)

    # plot tone analysis 
    document_tones = tones_doc1 + tones_doc2 + tones_doc3

    # Results_data[email]={
    #     "doc_tone":document_tones
    #     }

    satisfied_tone = []
    polite_tone = []
    excited_tone = []
    frustrated_tone = []
    sad_tone = []
    sympathetic_tone = []
    impolite_tone = []

    for sentiment, score in document_tones:
        if sentiment == "satisfied":
            satisfied_tone.append(score)
        elif sentiment == "polite":
            polite_tone.append(score)
        elif sentiment == "excited":
            excited_tone.append(score)
        elif sentiment == "frustrated":
            frustrated_tone.append(score)
        elif sentiment == "sad":
            sad_tone.append(score)
        elif sentiment == "sympathetic":
            sympathetic_tone.append(score)
        elif sentiment == "impolite":
            impolite_tone.append(score)

    values = np.array([0,1,2])*3
    fig = plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    plt.xlim(-1.5, 10)

    plt.bar(values , satisfied_tone , width = 0.4 , label = 'satisfied')
    plt.bar(values+0.4 , sad_tone , width = 0.4 , label = 'Confidence')  
    plt.bar(values+0.8 , excited_tone , width = 0.4 , label = 'excited')
    plt.bar(values-0.4 , frustrated_tone , width = 0.4 , label = 'frustrated')
    plt.bar(values-0.8 , polite_tone , width = 0.4 , label = 'polite')
    plt.xticks(ticks = values , labels = ['Question 1','Question 2','Question 3'] , fontsize = 15 , fontweight = 60)
    plt.yticks(fontsize = 12 , fontweight = 90)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')                    
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 5)
    plt.legend()
    plt.savefig(f'./static/tone_analysis.jpg' , bbox_inches = 'tight')

    # save all responses
    with open('./static/answers.json' , 'w') as file:
        json.dump(responses , file)

    # face emotion recognition - plotting the emotions against time in the video
    videos = ["question1.webm", "question2.webm", "question3.webm"]
   # Output file path
    output_file = './static/combined.webm'

    # Initialize variables
    frame_width, frame_height = None, None
    output_video = None

    # Iterate through input videos
    for video in videos:
        cap = cv2.VideoCapture(f'./static/{video}')

        # Check if the video file was successfully opened
        if not cap.isOpened():
            print(f"Error opening video file: {video}")
            continue

        # Retrieve the frame dimensions from the first video
        if frame_width is None or frame_height is None:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create the output video writer on the first iteration
        if output_video is None:
            fourcc = cv2.VideoWriter_fourcc(*"VP90")
            output_video = cv2.VideoWriter(output_file, fourcc, 60, (frame_width, frame_height))

        # Read and write each frame to the output video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output_video.write(frame)

        # Release the current video capture
        cap.release()

    # Release the output video writer
    output_video.release()

    print("Video combination complete!")
    try:
        check_malpractice()
    except Exception as e:
        print(str(e))
    face_detector = FER(mtcnn=True)
    video_path = "./static/combined.webm"
    input_video = Video(video_path)
    processing_data = input_video.analyze(face_detector, display=False, save_frames=False, save_video=False, annotate_frames=False, zip_images=False)
    vid_df = input_video.to_pandas(processing_data)
    vid_df = input_video.get_first_face(vid_df)
    vid_df = input_video.get_emotions(vid_df)
    pltfig = vid_df.plot(figsize=(12, 6), fontsize=12).get_figure()
    plt.legend(fontsize='large', loc=1)

    pltfig.savefig('./static/fer_output.png')

    print("Success")

    return "success"


# Interview completed response message
@app.route('/recorded')
def response():
    return render_template('recorded.html')

def check_malpractice():

    malpractice_done=False
    malpractice_data={}
    face_cascade = cv2.CascadeClassifier("./static/haarcascade_frontalface_default.xml")

    # Initialize variables
    prev_center = None
    frame_count = 0
    horizontal_movement_count = 0
    vertical_movement_count = 0

    # Open the video file
    cap = cv2.VideoCapture(f'./static/combined.webm')

    # Loop through each frame
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        # Break if we have reached the end of the video
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
        # Loop through each face
        for (x,y,w,h) in faces:
            # Calculate the center of the current face
            center = (x + w//2, y + h//2)
            
            # If this is the first face detected, just store the center position and continue
            if prev_center is None:
                prev_center = center
                continue
            
            # Calculate the horizontal and vertical distance between the current and previous face center positions
            dx = abs(center[0] - prev_center[0])
            dy = abs(center[1] - prev_center[1])
            
            # Check if there was significant horizontal movement
            if dx > 30: # You can adjust this threshold as needed
                horizontal_movement_count += 1
                print(f"Frame {frame_count}: Significant horizontal movement detected")
            
            # Check if there was significant vertical movement
            if dy > 30: # You can adjust this threshold as needed
                vertical_movement_count += 1
                print(f"Frame {frame_count}: Significant vertical movement detected")
            
            # Store the current center position as the previous center position for the next iteration
            prev_center = center
            

        # Increment the frame count
        frame_count += 1
        
    # Print the final counts
    print(f"Total frames: {frame_count}")
    print(f"Horizontal movement count: {horizontal_movement_count}")
    print(f"Vertical movement count: {vertical_movement_count}")
    if horizontal_movement_count>4 or vertical_movement_count>4 :
        malpractice_done=True
        malpractice_data["malpractice_detected"]=malpractice_done
    else:
        malpractice_data["malpractice_detected"]=malpractice_done
    with open('./static/results.json' , 'w') as file:
        json.dump(malpractice_data, file)

# Display results to interviewee
@app.route('/info')
def info():
    with open('./static/result.json' , 'r') as file:
        output = json.load(file)

    with open('./static/answers.json' , 'r') as file:
        answers = json.load(file)

    return render_template('result.html' , output = output , responses = answers)


# Send job confirmation mail to selected candidate
@app.route('/accept' , methods=['GET'])
def accept():

    with open('./static/result.json' , 'r') as file:
        output = json.load(file)
    
    name = output['Name']
    email = output['Email']
    position = "Software Development Engineer"

    msg = Message(f'Job Confirmation Letter', sender = MAIL_USERNAME, recipients = [email])
    msg.body = f"Dear {name},\n\n" + f"Thank you for taking the time to interview for the {position} position. We enjoyed getting to know you. We have completed all of our interviews.\n\n"+ f"I am pleased to inform you that we would like to offer you the {position} position. We believe your past experience and strong technical skills will be an asset to our organization. Your starting salary will be $15,000 per year with an anticipated start date of July 1.\n\n"+ f"The next step in the process is to set up meetings with our CEO, Rahul Dravid\n\n."+ f"Please respond to this email by June 23 to let us know if you would like to accept the SDE position.\n\n" + f"I look forward to hearing from you.\n\n"+ f"Sincerely,\n\n"+ f"Harsh Verma\nHuman Resources Director\nPhone: 555-555-1234\nEmail: feedbackmonitor123@gmail.com"
    mail.send(msg)

    return "success"

# Send mail to rejected candidate
@app.route('/reject' , methods=['GET'])
def reject():

    with open('./static/result.json' , 'r') as file:
        output = json.load(file)
    
    name = output['Name']
    email = output['Email']
    position = "Software Development Engineer"

    msg = Message(f'Your application to Smart Hire', sender = MAIL_USERNAME, recipients = [email])
    msg.body = f"Dear {name},\n\n" + f"Thank you for taking the time to consider Smart Hire. We wanted to let you know that we have chosen to move forward with a different candidate for the {position} position.\n\n"+ f"Our team was impressed by your skills and accomplishments. We think you could be a good fit for other future openings and will reach out again if we find a good match.\n\n"+ f"We wish you all the best in your job search and future professional endeavors.\n\n"+ f"Regards,\n\n"+ f"Harsh Verma\nHuman Resources Director\nPhone: 555-555-1234\nEmail: feedbackmonitor123@gmail.com"
    mail.send(msg)

    return "success"


if __name__ == '__main__':
    app.run()
