import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import os
import shutil
from recognizer import FacialRecognizer

class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition System")
        self.root.geometry("800x600")
        self.root.configure(bg="#2e2e2e")

        self.recognizer = FacialRecognizer()
        self.current_image_path = None
        self.current_frame = None

        self.setup_ui()

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="Facial Recognition Dashboard", font=("Helvetica", 20, "bold"), bg="#2e2e2e", fg="white")
        header.pack(pady=20)

        # Image Display Area
        self.image_label = tk.Label(self.root, bg="#1e1e1e", text="No Image Loaded", fg="gray", font=("Helvetica", 16))
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # Control Panel
        control_frame = tk.Frame(self.root, bg="#2e2e2e")
        control_frame.pack(fill=tk.X, pady=20)

        # Buttons
        btn_style = {"font": ("Helvetica", 12), "width": 20, "bg": "#4a90e2", "fg": "white", "relief": tk.FLAT, "activebackground": "#357abd"}

        btn_load = tk.Button(control_frame, text="Load Target Image", command=self.load_image, **btn_style)
        btn_load.pack(side=tk.LEFT, padx=30)

        btn_run = tk.Button(control_frame, text="Run Recognition", command=self.run_recognition, **btn_style)
        btn_run.pack(side=tk.LEFT, padx=30)

        btn_add = tk.Button(control_frame, text="Add to Data Bank", command=self.add_to_data_bank, **btn_style)
        btn_add.pack(side=tk.LEFT, padx=30)

    def display_image(self, image=None):
        if image is None and self.current_image_path:
            # Fix OpenCV unicode path issue by using numpy array decoding instead of imread directly
            import numpy as np
            with open(self.current_image_path, "rb") as f:
                img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is not None:
            # Check maximum dimensions
            max_w, max_h = 700, 400
            h, w = image.shape[:2]
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                image = cv2.resize(image, (int(w * scale), int(h * scale)))

            # Only convert to RGB if it came from CV2 logic recently (opencv uses BGR)
            # The recognizer output gives RGB already, but reading directly gives BGR
            # Let's just assume we keep things in RGB before sending to PIL.
            img_pil = Image.fromarray(image)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.image_label.configure(image=img_tk, text="")
            self.image_label.image = img_tk  # Keep a reference

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if file_path:
            self.current_image_path = file_path
            # Provide BGR to RGB conversion for newly loaded files here
            # Fix OpenCV unicode path issue by using numpy array decoding instead of imread directly
            import numpy as np
            try:
                with open(file_path, "rb") as f:
                    img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                   self.display_image(img)
                else:
                   messagebox.showerror("Error", "Could not load the image.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def run_recognition(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load a target image first.")
            return

        result_frame = self.recognizer.recognize_in_image(self.current_image_path)
        if result_frame is not None:
            # result_frame is returned as BGR from OpenCV drawing, need to display as RGB
            rgb_result = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            self.display_image(rgb_result)
        else:
            messagebox.showerror("Error", "Recognition failed. See console for details.")

    def add_to_data_bank(self):
        file_path = filedialog.askopenfilename(
            title="Select Known Person Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if not file_path:
            return

        name = simpledialog.askstring("Input", "Enter the person's name:", parent=self.root)
        if name:
            name = name.strip()
            if not name:
                messagebox.showwarning("Warning", "Name cannot be empty.")
                return

            # Sanitize name for filename
            clean_name = "".join([c if c.isalnum() else "_" for c in name])
            ext = os.path.splitext(file_path)[1]
            new_filename = f"{clean_name}{ext}"
            new_filepath = os.path.join(self.recognizer.data_bank_dir, new_filename)

            try:
                shutil.copy2(file_path, new_filepath)
                messagebox.showinfo("Success", f"Added {clean_name} to the data bank!")
                # Reload the databank
                self.recognizer.known_face_encodings = []
                self.recognizer.known_face_names = []
                self.recognizer.load_data_bank()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.mainloop()
