import face_recognition
import cv2
import numpy as np
import os

class FacialRecognizer:
    def __init__(self, data_bank_dir="data_bank", tolerance=0.5):
        self.data_bank_dir = data_bank_dir
        self.tolerance = tolerance # Lower is stricter (0.6 is default, 0.4-0.5 is better for security)
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

    def recognize_in_frame(self, frame, process_rescale=0.25):
        """Processes a single BGR frame and returns the frame with annotations."""
        try:
            if process_rescale != 1.0:
                small_frame = cv2.resize(frame, (0, 0), fx=process_rescale, fy=process_rescale)
            else:
                small_frame = frame

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
                name = "Unknown"
                distance = 1.0 # Max distance

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    distance = face_distances[best_match_index]
                    if matches[best_match_index]:
                        # Distance is lower than tolerance
                        name = self.known_face_names[best_match_index]
                        # Convert distance to a simple "Confidence Score" (%)
                        # 0.0 distance = 100% match, self.tolerance = 0% boundary
                        confidence = max(0, (1 - (distance / self.tolerance)) * 100)
                        name = f"{name} ({confidence:.0f}%)"

                # Scale back up face locations since the frame we detected in was scaled
                if process_rescale != 1.0:
                    inv_scale = 1.0 / process_rescale
                    top = int(top * inv_scale)
                    right = int(right * inv_scale)
                    bottom = int(bottom * inv_scale)
                    left = int(left * inv_scale)

                # Choose color: Green for recognized, Red for Unknown
                color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)
        except Exception as e:
            print(f"Error in recognition: {e}")

        return frame

    def recognize_in_image(self, image_path):
        """Processes a single image and attempts to recognize faces."""
        print(f"Processing target image: {image_path}")
        if not os.path.exists(image_path):
            print(f"Error: Target image '{image_path}' not found.")
            return

        # Load the image using OpenCV
        import numpy as np
        try:
            with open(image_path, "rb") as f:
                img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error reading the file: {e}")
            return
            
        if frame is None:
             print(f"Error: Could not decode image at {image_path}.")
             return

        # Perform recognition at FULL resolution for better accuracy on static images
        frame = self.recognize_in_frame(frame, process_rescale=1.0)

        # Display output only if called as script
        if __name__ == "__main__":
            # Downscale for display if image is too large
            height, width = frame.shape[:2]
            if max(height, width) > 1000:
                scale = 1000 / max(height, width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            cv2.imshow('Facial Recognition', frame)
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
