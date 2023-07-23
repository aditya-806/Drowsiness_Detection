from scipy.spatial import distance #used for measuring the distance 
from imutils import face_utils     #used for face recognition
import imutils           #used for making basic image processing functions
import dlib              #used for landmark's facial detection with pre-trained models
import cv2               #used for face and eye detection
from pygame import mixer #used for playing sound
mixer.init()
alarm_sound = mixer.Sound('alarm.wav')

def eye_aspect_ratio(eye):
    #compute the euclidean distances between two sets of vertical eye
	#landmarks (x,y) coordinates
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	
	#compute the euclidean distance between the horizontal eye 
	#landmark (x,y) coordinates
	C = distance.euclidean(eye[0], eye[3])
	#compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
    #return the EYE ASPECT RATIO
	return ear
	
#define 2 constants,one for eye aspect ratio to indicate  blink and 
#second for no. of consecutive frames eye must be below 
#threshold to set off the alarm
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

#grab the indexes of the facial landmarks for the left and right eye,respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

#start video streaming part
cap=cv2.VideoCapture(0)
flag=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		#converting to NumPy Array
		shape = face_utils.shape_to_np(shape)

		#extract the left and right eye coordinates then use the coordinates to compute
		#eye aspect ratio for both
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		#find average of both eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		#compute the convexhull for both the eyes and visualise each eye
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:
    				#display WOKE UP message on screen if the eyes are closed for sufficient no. of time
				cv2.putText(frame, "****************DETECTED****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************DETECTED****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
				#play sound of alarm if eyes are closed for sufficient no. of time
				alarm_sound.play()
				#print drowsy in the terminal if eyes are closed
				print ("Person Detected")
		else:
			flag = 0
			#stop salarm sound after opening of eyes
			alarm_sound.stop()
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	#press q for ending the loop
	if key == ord("q"):
		break
#stop all the processing
cv2.destroyAllWindows()
cap.release() 
