import cv2
import sys

# Get input video from system arguments
input_video = sys.argv[1]

# Select classifier for haar landamark detection
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Create video capture object to take video input and process first frame
cap = cv2.VideoCapture(input_video)
has_frame, frame = cap.read()

# Create video wrier object to create output video object
vid_writer = cv2.VideoWriter('temp-{}.avi'.format(str(input_video).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

counter = 0

while(1):
    has_frame,frame = cap.read()
    if not has_frame:
        break
    count += 1

    # Call function to give output frames
    outOpencvHaar = detectFaceOpenCVHaar(faceCascade, frame)

    # Show bounding box output
    cv2.imshow("Face Detection Comparison", outOpencvHaar)

    # Write output frames to video object created before
    vid_writer.write(outOpencvHaar)

    # Wait for 'q' key to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
vid_writer.release()
