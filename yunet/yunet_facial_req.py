# reference https://opencv.org/blog/opencv-face-detection-cascade-classifier-vs-yunet/#:~:text=Face%20Detection%20is%20a%20computer,with%20various%20models%20being%20developed.

#reference https://gist.github.com/UnaNancyOwen/3f06d4a0d04f3a75cc62563aafbac332

from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2

# initialize 'currentname' to trigger only when a new person is identified
currentname = "unknown"
# determine faces from encodings.pickle file model created from train_model.py
encodingsP = "../encodings/encodings.pickle"

# model = YuNet(modelPath='face_detection_yunet_2022mar.onnx',
#               inputSize=[320, 320],
#               confThreshold=0.9,
#               nmsThreshold=0.3,
#               topK=5000,
#               backendId=3,
#               targetId=0)

detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "",(0,0))
# detect returns 
# image	an image to detect
# faces	detection results stored in a 2D cv::Mat of shape [num_faces, 15]
# 0-1: x, y of box top left corner
# 2-3: width, height of bbox
# 4-5: x, y of right eye (blue point in the example image)
# 6-7: x, y of left eye (red point in the example image)
# 8-9: x, y of nose tip (green point in the example image)
# 10-11: x, y of right corner of mouth (pink point in the example image)
# 12-13: x, y of left corner of mouth (yellow point in the example image)
# 14: face score

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# print(data['names'])
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	height, width, _ = frame.shape
	detector.setInputSize((width, height))

	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Getting detections
	other,detections = detector.detect(rgb)

	detections = detections if detections is not None else []

	# print(detections)
	boxes = [list(map(int, d[:4])) for d in detections]
	box = [(y, x + w, y + h, x) for (x, y, w, h) in boxes]
	print(box)

	# box=boxes[0]

	# print(boxes[0])
	confidence = str([d[-1] for d in detections])[1:6]
	# print(boxes)
	
	encodings = face_recognition.face_encodings(rgb, box)
	names = []

	# loop over the facial embeddings

	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding,tolerance= 0.3)
		name = "Unknown" # if face is not recognized, then print Unknown

		# print(matches)
		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face
			for i in matchedIdxs:
				name = data["names"][i]
				# print(counts.get(name, 0))
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
			

			# if someone in your dataset is identified, print their name on the screen
			if currentname != name:
				currentname = name
				print(currentname,counts)
			print(counts)
		# update the list of names
		names.append(name)
		
	# loop over the recognized faces
	for ((left, top, width, height), name) in zip(boxes, names):
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame,(left, top), (left + width, top + height),(0,255,0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)
		cv2.putText(frame, confidence, (left,y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)


	# loop over the recognized faces
	# display the image to our screen
	cv2.imshow("Facial Recognition is Running", frame)
	key = cv2.waitKey(1) & 0xFF

	# quit when 'q' key is pressed
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
