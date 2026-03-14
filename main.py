import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Download the face detection model if not present
model_path = "blaze_face_short_range.tflite"


# Initialize video capture
capture = cv2.VideoCapture(0)

# Configure face detection
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.7)
detector = vision.FaceDetector.create_from_options(options)

while True:
    ret, frame = capture.read()


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    # Draw bounding boxes manually
    if results.detections:
        for detection in results.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw score
            score = detection.categories[0].score
            cv2.putText(frame, f"{score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
capture.release()
cv2.destroyAllWindows()