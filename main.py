import cv2

# Load the models
face_model = cv2.dnn.readNet('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
age_model = cv2.dnn.readNet('age_net.caffemodel', 'age_deploy.prototxt')
gender_model = cv2.dnn.readNet('gender_net.caffemodel', 'gender_deploy.prototxt')

# Define mean values and model input size
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']


# Function to load an image
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read the image.")
    return img


# Function to detect face, age, and gender
def detect_age_gender(image_path, output_path):
    img = load_image(image_path)
    if img is None:
        print("Image loading failed.")
        return

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    face_model.setInput(blob)
    detections = face_model.forward()

    print(f"Detections shape: {detections.shape}")
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        print(f"Detection {i} confidence: {confidence}")
        if confidence > 0.9:  # Increased confidence threshold
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            print(f"Face detected at [{x1}, {y1}, {x2}, {y2}]")

            face = img[y1:y2, x1:x2]
            if face.size == 0:
                print("Invalid face region detected, skipping.")
                continue

            # Predict age
            age_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            age_model.setInput(age_blob)
            age_preds = age_model.forward()
            age = age_list[age_preds[0].argmax()]
            print(f"Predicted age: {age}")

            # Predict gender
            gender_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_model.setInput(gender_blob)
            gender_preds = gender_model.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print(f"Predicted gender: {gender}")

            # Draw bounding box and write age and gender
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            overlay_text = f"{gender}, {age}"
            cv2.putText(img, overlay_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")


# Path to the image
image_path = 'faces/person1.jpg'
output_path = 'output/person1_output.jpg'
detect_age_gender(image_path, output_path)
