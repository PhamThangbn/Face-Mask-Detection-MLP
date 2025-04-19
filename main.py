import cv2
import numpy as np
import joblib
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from mtcnn import MTCNN

# Load mask classifier model
mask_classifier_model_path = "model/Model_MLP.pkl"
mask_classifier_model = joblib.load(mask_classifier_model_path)

# Initialize MTCNN
detector = MTCNN()

# Check and create output directory if it doesn't exist
output_dir = "output_videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class FaceMaskDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Mask Detector")
        self.root.geometry("800x600")

        # Variables for video capture and images
        self.cap = None
        self.is_running = False
        self.out = None
        self.image_list = []
        self.processed_images = []  # Store processed images
        self.current_image_index = 0
        self.display_index = 0  # Index of displayed image

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(pady=10, fill=tk.BOTH, expand=True)

        # Frame for control buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)

        # Button to select video
        self.btn_select_video = tk.Button(self.button_frame, text="Select Video", command=self.select_video)
        self.btn_select_video.pack(side=tk.LEFT, padx=5)

        # Button to select images
        self.btn_select_images = tk.Button(self.button_frame, text="Select Images", command=self.select_images)
        self.btn_select_images.pack(side=tk.LEFT, padx=5)

        # Button to use webcam
        self.btn_webcam = tk.Button(self.button_frame, text="Use Webcam", command=self.use_webcam)
        self.btn_webcam.pack(side=tk.LEFT, padx=5)

        # Stop button
        self.btn_stop = tk.Button(self.button_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

    def select_images(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Another function is running. Please stop it before selecting images!")
            return

        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if file_paths:
            # Clear existing tabs
            for tab in self.notebook.tabs():
                self.notebook.forget(tab)

            # Create images tab
            self.image_tab = tk.Frame(self.notebook)
            self.notebook.add(self.image_tab, text="Images")

            # Create and display navigation frame
            self.nav_frame = tk.Frame(self.image_tab)
            self.nav_frame.pack(pady=5)
            self.btn_prev = tk.Button(self.nav_frame, text="<", command=self.show_prev_image, state=tk.DISABLED)
            self.btn_prev.pack(side=tk.LEFT, padx=5)
            self.index_label = tk.Label(self.nav_frame, text="")
            self.index_label.pack(side=tk.LEFT, padx=5)
            self.btn_next = tk.Button(self.nav_frame, text=">", command=self.show_next_image, state=tk.DISABLED)
            self.btn_next.pack(side=tk.LEFT, padx=5)
            self.image_label = tk.Label(self.image_tab)
            self.image_label.pack(pady=10)

            self.image_list = list(file_paths)
            self.processed_images = []
            self.current_image_index = 0
            self.display_index = 0
            self.is_running = True
            self.btn_select_video.config(state=tk.DISABLED)
            self.btn_webcam.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            # Show navigation buttons only if there are 2 or more images
            if len(self.image_list) >= 2:
                self.btn_prev.config(state=tk.DISABLED)  # Will be enabled as needed
                self.btn_next.config(state=tk.DISABLED)  # Will be enabled as needed
            else:
                self.btn_prev.pack_forget()
                self.btn_next.pack_forget()
            self.process_images()

    def process_images(self):
        if not self.is_running or self.current_image_index >= len(self.image_list):
            # If stopped or processing complete, reset state
            self.is_running = False
            self.btn_select_video.config(state=tk.NORMAL)
            self.btn_select_images.config(state=tk.NORMAL)
            self.btn_webcam.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            if self.processed_images:
                self.display_index = min(self.display_index, len(self.processed_images) - 1)
                self.show_image()
            return

        # Read and process current image
        image_path = self.image_list[self.current_image_index]
        image = cv2.imread(image_path)
        if image is not None:
            # Convert to RGB for MTCNN processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Detect faces with MTCNN
            faces = detector.detect_faces(image_rgb)

            for face in faces:
                confidence = face['confidence']
                if confidence > 0.5:  # Filter high-confidence faces
                    box = face['box']  # [x, y, width, height]
                    x, y, width, height = box
                    x_end, y_end = x + width, y + height
                    face_region = image[y:y_end, x:x_end]
                    if face_region.size > 0:
                        face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                        face_region = cv2.resize(face_region, (128, 128))
                        face_region = np.expand_dims(face_region, axis=0)
                        face_region = face_region / 255.0
                        face_region = face_region.flatten()
                        mask_prediction = mask_classifier_model.predict(face_region.reshape(1, -1))
                        label = "Without mask" if mask_prediction[0] > 0.5 else "With mask"
                        color = (0, 0, 255) if mask_prediction[0] > 0.5 else (0, 255, 0)
                        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        cv2.rectangle(image, (x, y), (x_end, y_end), color, 2)

            # Store processed image
            self.processed_images.append(image)
            if self.current_image_index == 0:
                self.show_image()

        # Move to next image
        self.current_image_index += 1
        self.root.after(10, self.process_images)

    def show_image(self):
        if not self.processed_images:
            return

        # Display current image
        image = self.processed_images[self.display_index]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_rgb)
        img = img.resize((540, 400), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.configure(image=imgtk)
        self.image_label.image = imgtk  # Keep reference

        # Update <current/total> label
        self.index_label.config(text=f"{self.display_index + 1}/{len(self.image_list)}")

        # Update navigation button states (if 2 or more images)
        if len(self.image_list) >= 2:
            self.btn_prev.config(state=tk.NORMAL if self.display_index > 0 else tk.DISABLED)
            self.btn_next.config(state=tk.NORMAL if self.display_index < len(self.processed_images) - 1 else tk.DISABLED)

    def show_prev_image(self):
        if self.display_index > 0:
            self.display_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.display_index < len(self.processed_images) - 1:
            self.display_index += 1
            self.show_image()

    def select_video(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Another function is running. Please stop it before selecting a video!")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            # Clear existing tabs
            for tab in self.notebook.tabs():
                self.notebook.forget(tab)
            self.start_video(file_path)

    def use_webcam(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Another function is running. Please stop it before using the webcam!")
            return

        # Clear existing tabs
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        self.start_video(0)

    def start_video(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video or webcam!")
            return

        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_new = 25

        if source != 0:
            output_filename = os.path.splitext(os.path.basename(source))[0]
            output_filepath = os.path.join(output_dir, f"result_{output_filename}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_filepath, fourcc, fps_new, (self.original_width, self.original_height))

        # Create video tab
        self.video_tab = tk.Frame(self.notebook)
        self.notebook.add(self.video_tab, text="Video")
        self.display_frame = tk.Label(self.video_tab)
        self.display_frame.pack()

        self.is_running = True
        self.btn_select_images.config(state=tk.DISABLED)
        self.btn_webcam.config(state=tk.DISABLED)
        self.btn_select_video.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.process_video()

    def process_video(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        # Convert to RGB for MTCNN processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces with MTCNN
        faces = detector.detect_faces(frame_rgb)

        for face in faces:
            confidence = face['confidence']
            if confidence > 0.5:  # Filter high-confidence faces
                box = face['box']  # [x, y, width, height]
                x, y, width, height = box
                x_end, y_end = x + width, y + height
                face_region = frame[y:y_end, x:x_end]
                if face_region.size > 0:
                    face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                    face_region = cv2.resize(face_region, (128, 128))
                    face_region = np.expand_dims(face_region, axis=0)
                    face_region = face_region / 255.0
                    face_region = face_region.flatten()
                    mask_prediction = mask_classifier_model.predict(face_region.reshape(1, -1))
                    label = "Without mask" if mask_prediction[0] > 0.5 else "With mask"
                    color = (0, 0, 255) if mask_prediction[0] > 0.5 else (0, 255, 0)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.rectangle(frame, (x, y), (x_end, y_end), color, 2)

        if self.out is not None:
            self.out.write(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.display_frame.imgtk = imgtk
        self.display_frame.configure(image=imgtk)

        self.root.after(10, self.process_video)

    def stop(self):
        self.is_running = False
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_select_images.config(state=tk.NORMAL)
        self.btn_select_video.config(state=tk.NORMAL)
        self.btn_webcam.config(state=tk.NORMAL)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.out is not None:
            self.out.release()
            self.out = None
        if hasattr(self, 'video_tab'):
            self.notebook.forget(self.video_tab)
            delattr(self, 'video_tab')
        cv2.destroyAllWindows()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceMaskDetectorApp(root)
    app.run()