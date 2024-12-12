import cv2
import os
import tkinter as tk
from datetime import datetime
import face_recognition
import threading

# Load known face encodings (images from the 'user_images/' directory)
known_face_encodings = []
known_face_names = []

# Dictionary to track entry and exit times for each person
attendance_dict = {}

# Load all user images and encode their faces
user_images_path = 'project/user_images/'  # Ensure this path is correct
for file_name in os.listdir(user_images_path):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        image = face_recognition.load_image_file(f'{user_images_path}{file_name}')
        
        # Ensure face encodings are detected
        face_encoding = face_recognition.face_encodings(image)
        if face_encoding:  # Ensure a face was found in the image
            known_face_encodings.append(face_encoding[0])  # Get the first face encoding
            known_face_names.append(os.path.splitext(file_name)[0])
        else:
            print(f"Warning: No face found in image {file_name}")

def mark_attendance(name, entry_time, exit_time=None):
    csv_file_path = 'project/attendance.csv'  # Ensure this path is correct
    with open(csv_file_path, 'a') as f:
        if exit_time:
            f.write(f'{name},{entry_time},{exit_time}\n')
        else:
            f.write(f'{name},{entry_time},\n')

def show_gui():
    try:
        root = tk.Tk()
        root.title("Attendance System")
        root.mainloop()
    except Exception as e:
        print(f"Error starting Tkinter GUI: {e}")

def handle_multiple_faces(frame, face_locations, face_encodings):
    recognized_faces = set()  

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)  

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if name not in attendance_dict:
                attendance_dict[name] = {'entry_time': current_time, 'exit_time': None}
                mark_attendance(name, current_time) 

            recognized_faces.add(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return recognized_faces

def handle_exit(attendance_dict, recognized_faces, frame_count, max_unrecognized_frames=5):
    for name, times in list(attendance_dict.items()):
        if name not in recognized_faces:
            if times.get('unrecognized_frames', 0) >= max_unrecognized_frames:
                exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                attendance_dict[name]['exit_time'] = exit_time
                mark_attendance(name, times['entry_time'], exit_time)  
                del attendance_dict[name] 
            else:
                attendance_dict[name]['unrecognized_frames'] = times.get('unrecognized_frames', 0) + 1
        else:
            attendance_dict[name]['unrecognized_frames'] = 0

try:
    video_capture = cv2.VideoCapture(0)

    gui_thread = threading.Thread(target=show_gui, daemon=True)
    gui_thread.start()

    frame_count = 0  

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_locations:
            print("No faces detected in the frame.")
        
        recognized_faces = handle_multiple_faces(frame, face_locations, face_encodings)

        handle_exit(attendance_dict, recognized_faces, frame_count)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

except KeyboardInterrupt:
    print("Program interrupted. Exiting gracefully...")
finally:
    video_capture.release()
    cv2.destroyAllWindows()
