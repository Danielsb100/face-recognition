import face_recognition
import cv2
import numpy as np
import os

class FacialRecognizer:
    def __init__(self, data_bank_dir="data_bank"):
        self.data_bank_dir = data_bank_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_data_bank()

    def load_data_bank(self):
        """Loads images from the data bank and encodes them."""
        print(f"Loading known faces from '{self.data_bank_dir}'...")
        if not os.path.exists(self.data_bank_dir):
            os.makedirs(self.data_bank_dir)
            print(f"Directory '{self.data_bank_dir}' created. Please add images of known suspects.")
            return

        for filename in os.listdir(self.data_bank_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(self.data_bank_dir, filename)
                name = os.path.splitext(filename)[0] # e.g., "john_doe.jpg" -> "john_doe"
                
                # Load image and get encoding
                try:
                    image = face_recognition.load_image_file(filepath)
                    # Get the 128-dimension face encoding
                    # We assume each image has exactly one face, so we take the first encoding
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        encoding = encodings[0]
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(name)
                        print(f"Loaded: {name}")
                    else:
                        print(f"Warning: No face found in {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"Finished loading {len(self.known_face_names)} faces.")

    def recognize_in_image(self, image_path):
        """Processes a single image and attempts to recognize faces."""
        print(f"Processing target image: {image_path}")
        if not os.path.exists(image_path):
            print(f"Error: Target image '{image_path}' not found.")
            return

        # Load the image using OpenCV for display later, and face_recognition for processing
        # Use numpy logic to bypass OpenCV's utf-8 path-loading issues
        import numpy as np
        try:
            with open(image_path, "rb") as f:
                img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error reading the file: {e}")
            return
            
        if frame is None:
             print(f"Error: Could not decode image at {image_path}. Check if the file is valid.")
             return
             
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        print(f"Found {len(face_locations)} face(s) in this image.")

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            print(f"Detected: {name} at {(top, right, bottom, left)}")

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

        # Display the resulting image natively only if ran from command line
        if __name__ == "__main__":
            # Downscale for display if image is too large
            height, width = frame.shape[:2]
            if max(height, width) > 1000:
                scale = 1000 / max(height, width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            cv2.imshow('Facial Recognition', frame)
            print("Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return frame

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Facial Recognition System")
    parser.add_argument("--test-image", type=str, help="Path to a test image to run recognition on.")
    
    args = parser.parse_args()
    
    recognizer = FacialRecognizer()
    
    if args.test_image:
        recognizer.recognize_in_image(args.test_image)
    else:
        print("Setup complete. To test an image, use: python recognizer.py --test-image path/to/image.jpg")
