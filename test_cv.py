import cv2

# Load OpenCV's built-in face detector
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("Haar cascade loaded:", not cascade.empty())

# Open the video (path is relative to the HealthSync folder)
cap = cv2.VideoCapture("DATASET_2/subject1/vid.avi")
print("Video opened:", cap.isOpened())

# Read the first 10 frames and look for a face in each one
for i in range(10):
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    print(i, "faces found:", len(faces), flush=True)

cap.release()
print("TEST DONE")