import os
import sys
import cv2
import numpy as np
import pickle
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from deepface import DeepFace
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QMessageBox, QListWidget, QTabWidget, QLineEdit)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

class FaceRecognitionSystem:
    def __init__(self):
        # Face database
        self.face_db = {}
        self.embeddings = []
        self.labels = []
        self.nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        
        # Create necessary directories
        os.makedirs('face_db', exist_ok=True)
        os.makedirs('registered_faces', exist_ok=True)
        
        # Load existing database
        self.load_face_db()
        
    def load_face_db(self):
        """Load face database from disk"""
        if os.path.exists('face_db/face_db.pkl'):
            with open('face_db/face_db.pkl', 'rb') as f:
                self.face_db = pickle.load(f)
                
        if os.path.exists('face_db/embeddings.npy'):
            self.embeddings = np.load('face_db/embeddings.npy')
            
        if os.path.exists('face_db/labels.npy'):
            self.labels = np.load('face_db/labels.npy')
            
        if len(self.embeddings) > 0:
            self.nn_model.fit(self.embeddings)
    
    def save_face_db(self):
        """Save face database to disk"""
        with open('face_db/face_db.pkl', 'wb') as f:
            pickle.dump(self.face_db, f)
            
        if len(self.embeddings) > 0:
            np.save('face_db/embeddings.npy', self.embeddings)
            
        if len(self.labels) > 0:
            np.save('face_db/labels.npy', self.labels)
    
    def register_face(self, image, name):
        """Register a new face"""
        if name in self.face_db:
            raise ValueError(f"Name '{name}' already exists in database")
            
        # Save image temporarily for DeepFace processing
        temp_path = "temp_register.jpg"
        cv2.imwrite(temp_path, image)
        
        try:
            # Get face embedding using DeepFace
            embedding_obj = DeepFace.represent(
                img_path=temp_path,
                model_name="Facenet",
                enforce_detection=True
            )
            
            if len(embedding_obj) == 0:
                raise ValueError("No face detected in the image")
            if len(embedding_obj) > 1:
                raise ValueError("Multiple faces detected - please provide an image with a single face")
                
            embedding = embedding_obj[0]["embedding"]
            
            # Save face data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"registered_faces/{name}_{timestamp}.jpg"
            cv2.imwrite(filename, image)
            
            self.face_db[name] = {
                'embedding': embedding,
                'image_path': filename,
                'registration_date': timestamp
            }
            
            # Update NN model
            self.embeddings.append(embedding)
            self.labels.append(name)
            
            self.nn_model.fit(np.array(self.embeddings))
            self.save_face_db()
            
            return True
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def recognize_face(self, image):
        """Recognize faces in an image"""
        if len(self.face_db) == 0:
            raise ValueError("No faces registered in database")
            
        # Save image temporarily for DeepFace processing
        temp_path = "temp_recognize.jpg"
        cv2.imwrite(temp_path, image)
        
        try:
            # Detect faces and get embeddings
            face_objs = DeepFace.extract_faces(
                img_path=temp_path,
                target_size=(160, 160),
                detector_backend="opencv",
                enforce_detection=True,
                align=True
            )
            
            embeddings = DeepFace.represent(
                img_path=temp_path,
                model_name="Facenet",
                enforce_detection=False
            )
            
            results = []
            
            for face_obj, embedding_obj in zip(face_objs, embeddings):
                embedding = embedding_obj["embedding"]
                face_area = face_obj["facial_area"]
                
                distances, indices = self.nn_model.kneighbors([embedding])
                
                recognized_name = self.labels[indices[0][0]]
                distance = distances[0][0]
                confidence = 1 - distance
                
                x, y, w, h = face_area["x"], face_area["y"], face_area["w"], face_area["h"]
                bbox = (x, y, w, h)
                
                results.append({
                    'name': recognized_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'embedding': embedding
                })
                
            return results
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def delete_face(self, name):
        """Delete a registered face"""
        if name not in self.face_db:
            raise ValueError(f"Name '{name}' not found in database")
            
        # Remove from database
        os.remove(self.face_db[name]['image_path'])
        del self.face_db[name]
        
        # Update embeddings and labels
        new_embeddings = []
        new_labels = []
        
        for i, label in enumerate(self.labels):
            if label != name:
                new_embeddings.append(self.embeddings[i])
                new_labels.append(label)
                
        self.embeddings = new_embeddings
        self.labels = new_labels
        
        if len(self.embeddings) > 0:
            self.nn_model.fit(np.array(self.embeddings))
            
        self.save_face_db()
        
        return True

class FaceRecognitionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 1000, 700)
        
        # Initialize face recognition system
        self.fr_system = FaceRecognitionSystem()
        
        # Create main widgets
        self.create_widgets()
        
        # Setup camera
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_camera_on = False
        
        # Current frame for processing
        self.current_frame = None
        
    def create_widgets(self):
        """Create all UI widgets"""
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel - camera and controls
        left_panel = QVBoxLayout()
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        left_panel.addWidget(self.camera_label)
        
        # Camera controls
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.start_btn)
        
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.capture_face)
        self.capture_btn.setEnabled(False)
        control_layout.addWidget(self.capture_btn)
        
        left_panel.addLayout(control_layout)
        
        # Right panel - tabs for different functions
        right_panel = QTabWidget()
        
        # Register tab
        register_tab = QWidget()
        register_layout = QVBoxLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name for registration")
        register_layout.addWidget(self.name_input)
        
        self.browse_btn = QPushButton("Browse Image")
        self.browse_btn.clicked.connect(self.browse_image)
        register_layout.addWidget(self.browse_btn)
        
        self.register_btn = QPushButton("Register Face")
        self.register_btn.clicked.connect(self.register_face)
        register_layout.addWidget(self.register_btn)
        
        register_tab.setLayout(register_layout)
        right_panel.addTab(register_tab, "Register")
        
        # Database tab
        db_tab = QWidget()
        db_layout = QVBoxLayout()
        
        self.db_list = QListWidget()
        self.update_db_list()
        db_layout.addWidget(self.db_list)
        
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_face)
        db_layout.addWidget(self.delete_btn)
        
        db_tab.setLayout(db_layout)
        right_panel.addTab(db_tab, "Database")
        
        # Recognition tab
        recog_tab = QWidget()
        recog_layout = QVBoxLayout()
        
        self.recog_result = QLabel("Recognition results will appear here")
        self.recog_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recog_layout.addWidget(self.recog_result)
        
        self.recognize_btn = QPushButton("Recognize Faces")
        self.recognize_btn.clicked.connect(self.recognize_faces)
        recog_layout.addWidget(self.recognize_btn)
        
        recog_tab.setLayout(recog_layout)
        right_panel.addTab(recog_tab, "Recognition")
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 60)
        main_layout.addWidget(right_panel, 40)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.is_camera_on:
            self.timer.stop()
            self.camera.release()
            self.start_btn.setText("Start Camera")
            self.capture_btn.setEnabled(False)
        else:
            if not self.camera.isOpened():
                if not self.camera.open(0):
                    QMessageBox.critical(self, "Error", "Could not open camera")
                    return
                
            self.timer.start(20)
            self.start_btn.setText("Stop Camera")
            self.capture_btn.setEnabled(True)
            
        self.is_camera_on = not self.is_camera_on
        
    def update_frame(self):
        """Update camera frame"""
        ret, frame = self.camera.read()
        if ret:
            # Store the current frame
            self.current_frame = frame.copy()
            
            # Convert to RGB and display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(q_img))
            
    def capture_face(self):
        """Capture current frame for registration"""
        if self.current_frame is not None:
            self.name_input.setFocus()
            QMessageBox.information(self, "Success", "Frame captured for registration")
        else:
            QMessageBox.warning(self, "Error", "No frame available")
            
    def browse_image(self):
        """Browse for an image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
            
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.current_frame = image
                QMessageBox.information(self, "Success", "Image loaded successfully")
                return
                
        QMessageBox.warning(self, "Error", "Could not load image")
        
    def register_face(self):
        """Register a new face"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a name")
            return
            
        if self.current_frame is None:
            QMessageBox.warning(self, "Error", "Please capture or browse an image first")
            return
            
        try:
            self.fr_system.register_face(self.current_frame, name)
            QMessageBox.information(self, "Success", f"Face registered as {name}")
            self.name_input.clear()
            self.update_db_list()
            self.current_frame = None
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def update_db_list(self):
        """Update the database list widget"""
        self.db_list.clear()
        for name in self.fr_system.face_db:
            self.db_list.addItem(f"{name} - {self.fr_system.face_db[name]['registration_date']}")
            
    def delete_face(self):
        """Delete selected face from database"""
        selected = self.db_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Error", "Please select a face to delete")
            return
            
        name = selected.text().split(" - ")[0]
        reply = QMessageBox.question(
            self, "Confirm", f"Delete {name} from database?", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.fr_system.delete_face(name)
                self.update_db_list()
                QMessageBox.information(self, "Success", f"Deleted {name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                
    def recognize_faces(self):
        """Recognize faces in the current frame"""
        if self.current_frame is None:
            QMessageBox.warning(self, "Error", "No image available for recognition")
            return
            
        try:
            results = self.fr_system.recognize_face(self.current_frame)
            if not results:
                self.recog_result.setText("No faces recognized")
                return
                
            # Draw rectangles and names on the image
            display_image = self.current_frame.copy()
            for result in results:
                x, y, w, h = result['bbox']
                cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_image, 
                           f"{result['name']} ({result['confidence']:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 255, 0), 2)
            
            # Convert to RGB and display
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            h, w, ch = display_image.shape
            bytes_per_line = ch * w
            q_img = QImage(display_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Show results in the recognition tab
            self.recog_result.setPixmap(QPixmap.fromImage(q_img))
            
            # Also show text results
            result_text = "\n".join(
                f"{res['name']} (confidence: {res['confidence']:.2f})" 
                for res in results
            )
            QMessageBox.information(self, "Recognition Results", result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
                
    def closeEvent(self, event):
        """Clean up on window close"""
        if self.is_camera_on:
            self.timer.stop()
            self.camera.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle("Fusion")
    
    # Create a dark palette
    dark_palette = app.palette()
    dark_palette.setColor(dark_palette.ColorRole.Window, Qt.GlobalColor.darkGray)
    dark_palette.setColor(dark_palette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(dark_palette.ColorRole.Base, Qt.GlobalColor.darkGray)
    dark_palette.setColor(dark_palette.ColorRole.AlternateBase, Qt.GlobalColor.gray)
    dark_palette.setColor(dark_palette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(dark_palette.ColorRole.Button, Qt.GlobalColor.darkGray)
    dark_palette.setColor(dark_palette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(dark_palette)
    
    window = FaceRecognitionUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()