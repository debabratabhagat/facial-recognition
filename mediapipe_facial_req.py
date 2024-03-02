from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import time
import cv2
import mediapipe as mp
import face_recognition
# initialize 'currentname' to trigger only when a new person is identified
currentname = "unknown"
# determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"


mp_face_detection = mp.solutions.face_detection

facedetection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)

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
	frame_height, frame_width, _ = frame.shape

	
	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	predictions = facedetection.process(rgb)
	# print(predictions.detections[0].location_data.relative_bounding_box)
	
	if predictions.detections:
    # # looping through all the detection/faces 
		for d in predictions.detections:
        # extracting the relative bounding box data 
        # converting normalize coordinate into pixel coordinate
        # use numpy multiply function to multiple width and heigh of face to corresponding values.
			box = np.multiply(
				[
					d.location_data.relative_bounding_box.xmin,
                	d.location_data.relative_bounding_box.ymin,
                	d.location_data.relative_bounding_box.width,
                	d.location_data.relative_bounding_box.height,
            ],
            [frame_width, frame_height, frame_width, frame_height],
        ).astype(int)
        
	boxes = [box]
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in boxes]

	# print(boxes)

	# compute the facial embeddings for each face bounding box

	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# # loop over the facial embeddings

	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings

		matches = face_recognition.compare_faces(data["encodings"],
			encoding,tolerance= 0.30)
		


		# print(matches)
		
		name = "Unknown" # if face is not recognized, then print Unknown

		# print(matches)
		# check to see if we have found a match
		if True in matches:

			distance = face_recognition.face_distance(data["encodings"],encoding)
			
			# distance = (distance[0] * 100)
			# print(data['names'] )

			# print(distance)
			for (i,d) in enumerate(distance):
				print(data["names"][i],":",d)

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
		
		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)

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
