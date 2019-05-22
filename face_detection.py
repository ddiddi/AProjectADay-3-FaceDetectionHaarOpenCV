import cv2
import sys

def detect_face_openCV(face_classifier, frame, height=300,width=0):
    output_frame = frame.copy()
    frame_height = output_frame.shape[0]
    frame_width = output_frame.shape[1]

    if not width:
        width = int((frame_width/frame_height)*height)

    # Create scale variables to comensate for classifer and image dimensions
    scale_height = frame_height/height
    scale_width = frame_width/width
    scale_frame = cv2.resize(output_frame, (width,height))

    # Convert frame to gray for face detection classifier input
    gray_frame = cv2.cvtColor(scale_frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray_frame)

    # Put box at face detection areas
    for (x,y,w,h) in faces:
        rect = [int(x*scale_width),int(y*scale_height),int((x+w)*scale_width),int((y+h)*scale_height)]
        cv2.rectangle(output_frame,(rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0),
                      int(round(frame_height / 150)), 4)
    return output_frame

# Get input video from system arguments
input_video = sys.argv[1]

# Select classifier for haar landamark detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create video capture object to take video input and process first frame
cap = cv2.VideoCapture(input_video)
has_frame, frame = cap.read()

# Create video wrier object to create output video object
vid_writer = cv2.VideoWriter('output-{}.avi'.format(str(input_video).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

while(1):
    has_frame,frame = cap.read()
    if not has_frame:
        break

    # Call function to give output frames
    outOpencvHaar = detect_face_openCV(face_classifier, frame)

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
