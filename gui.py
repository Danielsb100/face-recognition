import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import cv2
import os
import shutil
import threading
import queue
import time
from recognizer import FacialRecognizer

class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition System")
        self.root.geometry("800x650")
        self.root.configure(bg="#2e2e2e")

        self.recognizer = FacialRecognizer(tolerance=0.45) # Stricter than default
        self.current_image_path = None
        
        # Camera Threading State
        self.camera_active = False
        self.camera_thread = None
        self.display_queue = queue.Queue(maxsize=2)
        self.selected_camera_index = tk.IntVar(value=0)
        
        # General Processing State
        self.is_processing_manual = False
        
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="Facial Recognition Dashboard", font=("Helvetica", 20, "bold"), bg="#2e2e2e", fg="white")
        header.pack(pady=20)

        # Image Display Area
        self.image_label = tk.Label(self.root, bg="#1e1e1e", text="No Image Loaded", fg="gray", font=("Helvetica", 16))
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#3e3e3e", fg="lightgray")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Control Panel
        control_frame = tk.Frame(self.root, bg="#2e2e2e")
        control_frame.pack(fill=tk.X, pady=10)

        # Camera Selector Frame (Above buttons)
        cam_frame = tk.Frame(control_frame, bg="#2e2e2e")
        cam_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(cam_frame, text="Camera Index:", bg="#2e2e2e", fg="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(15, 5))
        self.cam_spinbox = ttk.Spinbox(cam_frame, from_=0, to=10, textvariable=self.selected_camera_index, width=5)
        self.cam_spinbox.pack(side=tk.LEFT)
        tk.Label(cam_frame, text="(Try 0, 1, or 2 if Brio is not showing)", bg="#2e2e2e", fg="gray", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=5)

        # Buttons Frame
        btn_frame = tk.Frame(control_frame, bg="#2e2e2e")
        btn_frame.pack(fill=tk.X)

        btn_style = {"font": ("Helvetica", 12), "width": 15, "bg": "#4a90e2", "fg": "white", "relief": tk.FLAT, "activebackground": "#357abd"}

        btn_load = tk.Button(btn_frame, text="Load Image", command=self.load_image, **btn_style)
        btn_load.pack(side=tk.LEFT, padx=15)

        self.btn_camera = tk.Button(btn_frame, text="Start Camera", command=self.toggle_camera, **btn_style)
        self.btn_camera.pack(side=tk.LEFT, padx=15)

        self.btn_run = tk.Button(btn_frame, text="Run Filter", command=self.run_manual_recognition, **btn_style)
        self.btn_run.pack(side=tk.LEFT, padx=15)

        btn_add = tk.Button(btn_frame, text="Add Person", command=self.add_to_data_bank, **btn_style)
        btn_add.pack(side=tk.LEFT, padx=15)

    def set_status(self, msg):
        self.status_var.set(msg)

    def display_image(self, image=None):
        """Displays an RGB image on the label."""
        if image is not None:
            max_w, max_h = 700, 400
            h, w = image.shape[:2]
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                image = cv2.resize(image, (int(w * scale), int(h * scale)))

            img_pil = Image.fromarray(image)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.image_label.configure(image=img_tk, text="")
            self.image_label.image = img_tk

    def _try_open_camera(self, index):
        """Attempts to open a camera with different backends."""
        # Use DSHOW first for Windows as MSMF tends to block indefinitely on high-end webcams like Logitech Brio
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"Successfully opened camera {index} with backend {backend}")
                    return cap
                cap.release()
        return None

    def camera_worker(self):
        """Background thread for camera capture and recognition."""
        cam_idx = self.selected_camera_index.get()
        self.root.after(0, lambda: self.set_status(f"Connecting to camera {cam_idx}..."))
        
        cap = self._try_open_camera(cam_idx)
        
        if not cap:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Could not open camera at index {cam_idx}.\nTry changing the index or check connections."))
            self.camera_active = False
            self.root.after(0, self.reset_camera_button)
            return

        self.root.after(0, lambda: self.set_status(f"Camera Active (Index {cam_idx})"))

        frame_count = 0
        last_processed_frame = None

        while self.camera_active:
            ret, frame = cap.read()
            if not ret:
                self.root.after(0, lambda: self.set_status("Camera feed lost."))
                break

            # Process every 5th frame for recognition, otherwise use last result
            if frame_count % 5 == 0:
                last_processed_frame = self.recognizer.recognize_in_frame(frame.copy(), process_rescale=0.25)
            
            frame_count += 1
            
            # Use the processed frame or the original if nothing yet
            display_frame = last_processed_frame if last_processed_frame is not None else frame
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Send to display queue
            if self.display_queue.full():
                try: self.display_queue.get_nowait()
                except: pass
            self.display_queue.put(rgb_frame)
            
            time.sleep(0.01) # Small sleep to reduce CPU usage

        cap.release()
        self.root.after(0, self.reset_camera_button)

    def reset_camera_button(self):
        self.btn_camera.configure(text="Start Camera", bg="#4a90e2", activebackground="#357abd")
        self.cam_spinbox.configure(state="normal")
        if not self.camera_active:
            self.set_status("Ready")

    def toggle_camera(self):
        if not self.camera_active:
            if self.is_processing_manual:
                messagebox.showwarning("Warning", "Please wait for manual recognition to finish.")
                return
            
            self.camera_active = True
            self.btn_camera.configure(text="Stop Camera", bg="#e74c3c", activebackground="#c0392b")
            self.cam_spinbox.configure(state="disabled")
            
            # Clear display queue
            while not self.display_queue.empty(): self.display_queue.get()
            
            # Start worker thread
            self.camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
            self.camera_thread.start()
            self.update_gui_loop()
        else:
            self.camera_active = False
            self.set_status("Stopping camera...")

    def update_gui_loop(self):
        """Main loop updates the UI from the display queue."""
        if self.camera_active:
            try:
                rgb_frame = self.display_queue.get_nowait()
                self.display_image(rgb_frame)
            except queue.Empty:
                pass
            self.root.after(10, self.update_gui_loop)
        else:
            self.image_label.configure(image="", text="Camera Stopped")

    def run_manual_recognition(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        if self.camera_active:
            messagebox.showwarning("Warning", "Please stop the camera first.")
            return

        if self.is_processing_manual:
            return

        self.is_processing_manual = True
        self.btn_run.configure(state=tk.DISABLED, text="Processing...")
        self.set_status("Running manual recognition (Full Quality)...")
        
        # Run in thread so GUI doesn't freeze
        threading.Thread(target=self._manual_recognition_thread, args=(self.current_image_path,), daemon=True).start()

    def _manual_recognition_thread(self, path):
        try:
            result_frame = self.recognizer.recognize_in_image(path)
            if result_frame is not None:
                rgb_result = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                self.root.after(0, lambda: self.display_image(rgb_result))
                self.root.after(0, lambda: self.set_status("Recognition Complete"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Recognition failed."))
        except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Processing error: {e}"))
        finally:
            self.root.after(0, self._manual_recognition_done)

    def _manual_recognition_done(self):
        self.is_processing_manual = False
        self.btn_run.configure(state=tk.NORMAL, text="Run Filter")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if file_path:
            self.current_image_path = file_path
            import numpy as np
            try:
                with open(file_path, "rb") as f:
                    img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                   self.display_image(img)
                   self.set_status(f"Loaded: {os.path.basename(file_path)}")
                else:
                   messagebox.showerror("Error", "Could not load image.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def add_to_data_bank(self):
        file_path = filedialog.askopenfilename(
            title="Select Known Person Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if not file_path: return

        name = simpledialog.askstring("Input", "Enter the person's name:", parent=self.root)
        if name:
            name = name.strip()
            if not name: return

            clean_name = "".join([c if c.isalnum() else "_" for c in name])
            ext = os.path.splitext(file_path)[1]
            new_filename = f"{clean_name}{ext}"
            new_filepath = os.path.join(self.recognizer.data_bank_dir, new_filename)

            try:
                shutil.copy2(file_path, new_filepath)
                messagebox.showinfo("Success", f"Added {clean_name}!")
                self.recognizer.known_face_encodings = []
                self.recognizer.known_face_names = []
                self.recognizer.load_data_bank()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def on_closing(self):
        self.camera_active = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.mainloop()
