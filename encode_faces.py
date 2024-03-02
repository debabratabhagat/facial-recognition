from imutils import paths
import face_recognition
import pickle
import cv2
import os

# import mediapipe as mp
# import numpy as np

# mp_face_detection = mp.solutions.face_detection

# facedetection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)



# mp_face_detection = mp.solutions.face_detection

# facedetection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)

# our images are located in the dataset folder
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))

# img = cv2.imread(imagePaths[-1])
# print(img.shape)

# image = cv2.imread(imagePaths[-1])
# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# print(image.shape)

# plt.subplot(1,2,1)
# plt.title("original img ")
# plt.imshow(image)

# plt.subplot(1,2,2)
# plt.title("converted img (RGB) ")
# plt.imshow(rgb)
# plt.show()

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
undetectable = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):

    # print(f"run : {i} -- path : {imagePath}")
	# extract the person name from the image path

	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	
	name = imagePath.split(os.path.sep)[1]
	# print(name)

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB) 
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	frame_height, frame_width, _ = rgb.shape
	# predictions = facedetection.process(rgb)
	# print(predictions.detections[0].location_data.relative_bounding_box)
	
	# if predictions.detections:
    # # # looping through all the detection/faces 
	# 	for d in predictions.detections:
    #     # extracting the relative bounding box data 
    #     # converting normalize coordinate into pixel coordinate
    #     # use numpy multiply function to multiple width and heigh of face to corresponding values.
	# 		box = np.multiply(
	# 			[
	# 				d.location_data.relative_bounding_box.xmin,
    #             	d.location_data.relative_bounding_box.ymin,
    #             	d.location_data.relative_bounding_box.width,
    #             	d.location_data.relative_bounding_box.height,
    #         ],
    #         [frame_width, frame_height, frame_width, frame_height],
    #     ).astype(int)
        
	# boxes = [box]

	# boxes = [(y, x + w, y + h, x) for (x, y, w, h) in boxes]
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	
	# boxes = face_recognition.face_locations(rgb,
	# 	model="hog")

	boxes = face_recognition.face_locations(rgb, model='HOG',number_of_times_to_upsample=2)


	# print(boxes)
	#returns a list containing the [top right bottom left] coordinates
	#some images dont have any faces thus boxes is empty
	#some show false detection thus have two lists of boxes
	
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	if len(encodings) == 0:
		undetectable.append(i+1)
	# if len(encodings) != 0:
	# 	print(len(encodings[0]))
	
	#The above line returns an array of 128
	#And an empty list[] if no face is detected
	#If the list is empty then the loop is skipped
	#And the next image is processed

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		# print("encoding : ", encoding) ## if no face is detected the encno
		knownEncodings.append(encoding)
		knownNames.append(name)

print("know encodings : ", len(knownEncodings))
print("know names : ", len(knownNames))
# for ele in undetectable:
# 	print(f"PIC : {knownNames[ele]} ", ele)
# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("./encodings/encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
