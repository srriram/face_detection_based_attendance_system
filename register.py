import cv2
import numpy as np
import mediapipe as mp
import os
from deepface import DeepFace

class register_student:
    def __init__(self, name, roll_no):
        self.name = name  
        self.roll_no = roll_no  
        self.counter = 0  # Counter to track image numbers
        self.directory = f"{self.name}_{self.roll_no}"
        
    def open_cam(self, ip_address):
        ip_cam = f"http://{ip_address}:8080/video"
        cam = cv2.VideoCapture(ip_cam)

        if not cam.isOpened():
            print("Error opening IP camera, switching to laptop camera.")
            cam = cv2.VideoCapture(0)
        
        mp_face_mesh = mp.solutions.face_mesh
        mp_face_detection = mp.solutions.face_detection

        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to access camera")
                break  

            original_frame = frame.copy()  
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    results_mesh = face_mesh.process(frame_rgb)

                    if results_mesh.multi_face_landmarks:
                        for face_landmarks in results_mesh.multi_face_landmarks:
                            left_eye = face_landmarks.landmark[33]
                            right_eye = face_landmarks.landmark[263]

                            dx = right_eye.x - left_eye.x
                            dy = right_eye.y - left_eye.y
                            angle = np.degrees(np.arctan2(dy, dx))

                            color = (0, 255, 0) if abs(angle) < 10 else (0, 0, 255)
                            text = "Face is Aligned" if abs(angle) < 10 else "Face is Not Aligned"

                            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("REGISTER STUDENT", frame)

            key = cv2.waitKey(1) & 0xFF  
            
            if key == ord('c'):  
                self.take_photos(original_frame)  
            elif key == ord('q'):  
                break  

        cam.release()
        cv2.destroyAllWindows()
    
    def take_photos(self, frame):
        if not os.path.exists(self.directory):
            try:
                os.mkdir(self.directory)
                print(f"Directory '{self.directory}' created successfully.")
            except Exception as e:
                print(f"Error creating directory: {e}")

        photo_path = os.path.join(self.directory, f"{self.name}_{self.counter}.png")
        cv2.imwrite(photo_path, frame)
        print(f"Photo saved at: {photo_path}")

        self.counter += 1  


class Attendance:
    def open_cam(self, ip_address):
        ip_cam = f"http://{ip_address}:8080/video"
        cam = cv2.VideoCapture(ip_cam)

        if not cam.isOpened():
            print("Error opening IP camera, switching to laptop camera.")
            cam = cv2.VideoCapture(0)

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to access camera")
                break  

            cv2.imshow("ATTENDANCE SYSTEM", frame)

            key = cv2.waitKey(1) & 0xFF  
            
            if key == ord('c'):  
                self.check_attendance(frame)
                break
            elif key == ord('q'):  
                break  

        cam.release()
        cv2.destroyAllWindows()

    def check_attendance(self, frame):
        temp_image_path = "temp_face.png"
        cv2.imwrite(temp_image_path, frame)

        registered_students = [d for d in os.listdir() if os.path.isdir(d)]

        found = False
        for student_folder in registered_students:
            student_images = [img for img in os.listdir(student_folder) if img.endswith(".png")]
            
            for student_img in student_images:
                student_image_path = os.path.join(student_folder, student_img)
                try:
                    result = DeepFace.verify(temp_image_path, student_image_path, model_name="Facenet", enforce_detection=False)
                    if result["verified"]:
                        print(f"{student_folder.split('_')[0]} is Present")
                        found = True
                        break
                except Exception as e:
                    print(f"Error processing {student_folder}: {e}")

            if found:
                break

        if not found:
            print("Face not recognized. Not registered.")

        os.remove(temp_image_path)


# Example Usage:
# student=register_student(name="student",roll_no="roll_no")
# student.open_cam(ip_address="address")
# attendance = Attendance()
# attendance.open_cam("address")

