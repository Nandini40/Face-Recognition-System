# Face-Recognition-System
A modern, user-friendly face recognition application with registration, database management, and real-time recognition capabilities.

markdown
# Face Recognition System with PyQt6 GUI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange)
![PyQt6](https://img.shields.io/badge/PyQt6-6.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A modern, user-friendly face recognition application with registration, database management, and real-time recognition capabilities.

## Features

- 🎥 **Real-Time Face Detection** - Webcam and image file support
- 📝 **Face Registration** - Save faces with names to a local database
- 🔍 **Face Recognition** - Identify registered faces with confidence scores
- 🗃️ **Database Management** - View, add, and delete registered faces
- 🖥️ **Modern GUI** - Clean PyQt6 interface with dark theme
- 🚀 **Easy Installation** - No complex dependencies like dlib


Usage
Register New Faces:
Click "Start Camera" to enable webcam
Capture an image with "Capture" button
Enter a name and click "Register Face"

Recognize Faces:
Click "Recognize Faces" to identify people
Results show with bounding boxes and confidence scores

Manage Database:
View all registered faces in the Database tab
Delete entries with "Delete Selected" button

Technical Details
Face Detection: OpenCV Haar Cascades
Face Recognition: DeepFace with Facenet model
Database: Local storage with pickle/numpy
UI Framework: PyQt6

Future Improvements
Add face clustering functionality
Support for cloud storage integration
Multi-face recognition in single image
Export/import database feature

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository
Create a new branch (git checkout -b feature/your-feature)
Commit your changes (git commit -m 'Add some feature')
Push to the branch (git push origin feature/your-feature)
Open a Pull Request



### Additional Recommendations:

1. **Add Screenshots:**
   - Create a `/screenshots` folder
   - Include images of:
     - Main interface
     - Registration process
     - Recognition results
     - Database view

2. **Create requirements.txt:**
   ```text
   deepface>=0.0.79
   opencv-python>=4.5.5
   numpy>=1.21.0
   scikit-learn>=1.0.0
   PyQt6>=6.2.0
