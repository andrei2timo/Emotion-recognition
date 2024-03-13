import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk, ImageDraw
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from load_and_process import preprocess_input
import numpy as np
from collections import defaultdict

class EmotionAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")
        self.root.geometry("380x300")  # initial size of the GUI

        # create the GUI
        self.main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # create the title in the main GUI
        self.title_label = ttk.Label(self.main_frame, text="Emotion Detector", font=('Arial', 16, 'bold'), foreground='#B85042')
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Analyze from picture button
        self.analyze_from_picture_btn = ttk.Button(self.main_frame, text="Analyze from Picture", command=self.analyze_from_picture, style='Custom.TButton')
        self.analyze_from_picture_btn.grid(row=1, column=0, padx=10, pady=10)

        # Analyze from camera button
        self.analyze_from_camera_btn = ttk.Button(self.main_frame, text="Analyze from Camera", command=self.analyze_from_camera, style='Custom.TButton')
        self.analyze_from_camera_btn.grid(row=1, column=1, padx=10, pady=10)

        # Analyse from camera - camera label
        self.camera_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.camera_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Camera label
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Camera frame for emotion detector
        self.sentiment_label = ttk.Label(self.camera_frame)
        self.sentiment_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Exit button
        self.exit_btn = ttk.Button(self.main_frame, text="Exit", command=self.root.quit, style='Custom.TButton')
        self.exit_btn.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        # load the trained model
        self.face_detection = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
        self.emotion_classifier = load_model('models/_mini_XCEPTION.102-0.66.hdf5', compile=False)
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

        # Create the syles for buttons
        self.style = ttk.Style()
        self.style.configure('Custom.TButton', foreground='black', font=('Arial', 12, 'bold'))
        self.style.map('Custom.TButton', background=[('pressed', '#E7E8D1'), ('active', '#E7E8D1')])


    """
        Function to analyze emotions from an image file.

        This function opens a file dialog to select an image file, reads the image,
        analyzes the emotions in the image using another method called analyze_image_from_picture,
        and displays the emotions in a table using a method called display_emotions_table.

        :return: None
        """
    def analyze_from_picture(self):
       # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        
        if not file_path:
            return  # User cancelled, do nothing
        
        # Read the image
        image = cv2.imread(file_path)
        
        # Analyze the emotions in the image
        emotions = self.analyze_image_from_picture(image)

    """
    Function to analyze emotions from an image.

    This function performs emotion analysis on the given image. It detects faces,
    predicts emotions for each detected face, and visualizes the results by
    overlaying emojis on the faces, drawing rectangles around the faces, and
    displaying the detected emotions as text.

    :param image: The input image for emotion analysis.
    :return: A dictionary containing the count of each detected emotion and the total number of faces.
    """
    def analyze_image_from_picture(self, image):
        emotions = defaultdict(int)
        total_faces = 0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        # Create a copy of the original image to draw on
        image_with_emotion = image.copy()
        
        # Based on the current position of the face, try to detect the depicted image
        for (fX, fY, fW, fH) in faces:
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)

            preds = self.emotion_classifier.predict(roi)[0]
            label = self.EMOTIONS[np.argmax(preds)]
            
            emotions[label] += 1
            total_faces += 1
            
            # Load and resize emoji
            emoji_filename = f"emojis/{label}.png"
            emoji_image = cv2.imread(emoji_filename)

            # Resize the emoji to the desired size
            emoji_image = cv2.resize(emoji_image, (20, 20))

            # Paste emoji at the static position
            emoji_position = (fX + fW - 20, fY)  # Adjusted position (top right corner of the face)
            image_with_emotion[emoji_position[1]:emoji_position[1] + 20, emoji_position[0]:emoji_position[0] + 20] = emoji_image

            # Draw a rectangle around the face
            cv2.rectangle(image_with_emotion, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 1)

            # Display the detected emotion text
            text_width, text_height = cv2.getTextSize(f"Emotion detected: {label}", cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)[0]
            text_position = (fX + (fW - text_width) // 2, fY - 10)  # Adjusted position for center alignment
            font_scale = 0.6  # Adjust the font scale for smaller text
            cv2.putText(image_with_emotion, f"Emotion detected: {label}", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

        # Adjust the frame size
        image_with_emotion = cv2.resize(image_with_emotion, (500, 500))

        # Exit the frame
        cv2.imshow('Emotion Detection from Picture', image_with_emotion)
        cv2.waitKey(0)  # Wait for any key to be pressed
        cv2.destroyAllWindows()


    """ Analyze from camera methods"""

    """
    Function to analyze emotions from a live camera feed.

    This function initializes the camera capture object, reads frames continuously,
    analyzes emotions from each frame using another method called analyze_image,
    and displays the camera feed in a label on the GUI window.

    :return: None
    """
    def analyze_from_camera(self):
        # Create the camera capture object
        self.camera = cv2.VideoCapture(0)

        while True:
            ret, frame = self.camera.read()

            if not ret:
                break

            # Analyze the emotion from the frame
            self.analyze_image(frame)

            # Display the camera feed in the camera label
            self.camera_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.camera_photo = ImageTk.PhotoImage(image=Image.fromarray(self.camera_image))
            self.camera_label.config(image=self.camera_photo)
            self.camera_label.image = self.camera_photo

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key == 27:  # Press 'Q' to exit
                self.root.quit()  # Quit the GUI application

        self.camera.release()
        cv2.destroyAllWindows()
        self.root.quit()  # Quit the GUI application
        
    """
    Function to analyze emotions from a single frame.

    This function resizes the frame, detects faces using a face detection model,
    analyzes emotions for each detected face, and displays the emotions along with
    rectangles around the faces and sentiment analysis in a label on the GUI window.

    :param frame: The frame from the camera feed
    :return: None
    """
    def analyze_image(self, frame):
        frame = cv2.resize(frame, (400, 400))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        emoji_size = (30, 30)  # Set the size of the emoji
        emoji_position = (frame.shape[1] - emoji_size[0] - 10, 10)  # Set the position of the emoji (top right corner)

        # based on the current position of the face, try to detect the depicted emotion (saddness, happiness etc)
        if len(faces) > 0:
            for (fX, fY, fW, fH) in faces:
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = self.emotion_classifier.predict(roi)[0]
                label = self.EMOTIONS[np.argmax(preds)]

                # Display the sentiment analysis in the sentiment label
                self.sentiment_label.config(text=label)

                # Load and resize emoji
                emoji_filename = f"emojis/{label}.png"
                emoji_image = cv2.imread(emoji_filename)

                # Resize the emoji to the desired size
                emoji_image = cv2.resize(emoji_image, emoji_size)

                # Paste emoji at the static position
                frame[emoji_position[1]:emoji_position[1] + emoji_size[1], emoji_position[0]:emoji_position[0] + emoji_size[0]] = emoji_image

                # Draw a rectangle around the face
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

                # Display the detected emotion text
                text_position = (fX, fY - 10)  # Adjusted position
                font_scale = 0.6  # Adjust the font scale for smaller text
                cv2.putText(frame, f"Emotion detected: {label}", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

            cv2.imshow('Emotion Detection', frame)
        else:
            cv2.imshow('Emotion Detection', frame)


# Create the main window
root = tk.Tk()
app = EmotionAnalyzerApp(root) # Call the function and start the app
root.mainloop()
