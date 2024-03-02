import cv2
import sys
import uuid
import os 

# print(type(sys.argv))
name = input("Enter the name : ") # name passed from command line argument

currentpath = f"A:\\facial_recognition\\facial-recognition\dataset\{name}"

# print(os.path.exists(currentpath))

if not os.path.exists(currentpath):
    os.mkdir(currentpath)


cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("press space to take a photo", 500, 300)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    if not ret:
        print("failed to grab frame")
        break

    cv2.imshow("press space to take a photo", frame)
    k = cv2.waitKey(1)

    if k & 0xFF == ord('q'):
        # q pressed
        print("closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "dataset/{}/image_{}.jpg".format(name, uuid.uuid4().hex)
        # print(img_name)
        status = cv2.imwrite(img_name, frame)
        if status is True:
            print("{} written!".format(img_name))
        else:
            print("Image not written. Check person's folder created")

cam.release()
cv2.destroyAllWindows()
