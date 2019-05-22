import cv2
import sys

# Get input video from system arguments
input_video = sys.argv[1]

# Select classifier for haar landamark detection
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Create video capture object to take video input and process into frames
cap = cv2.VideoCapture(input_video)
has_frame, frame = cap.read()

# Create video wrier object to create output video object
vid_writer = cv2.VideoWriter('temp-{}.avi'.format(str(input_video).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))
