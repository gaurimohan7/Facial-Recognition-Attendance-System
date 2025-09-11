import pandas as pd
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

print("To stop the webcam, please press q")

#Step 1: Load dataset images

path = "C:/Users/shobh/OneDrive/Face recognition photos"
images = []
classNames = []
myList = os.listdir(path)
print("Found dataset images:", myList)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    if curImg is None:
        print(f"[SKIP] {cl} could not be read")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # remove extension
print("Class names:", classNames)


#Step 2: Encode known faces

def findEncodings(images):
    encodeList = []
    for idx, img in enumerate(images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(img_rgb)
        if encs:
            encodeList.append(encs[0])
            print(f"[OK] Encoded {classNames[idx]}")
        else:
            print(f"[WARN] No face found in {classNames[idx]} — skipping")
    return encodeList
encodeListKnown = findEncodings(images)
print("Encoding complete! Total encodings:", len(encodeListKnown))


#Step 3: Attendance logging

def markAttendance(name):
    if not os.path.exists("Attendance.csv"):
        with open("Attendance.csv", "w") as f:
            f.write("Name,Time\n")

    with open("Attendance.csv", "r+") as f:
        dataList = f.readlines()
        nameList = [line.split(",")[0] for line in dataList]

        now = datetime.now()
        dtString = now.strftime("%H:%M:%S")

        if name == "UNKNOWN":
            f.write(f"{name},{dtString}\n")  # log always
            print(f"[LOG] Unknown detected at {dtString}")
        else:
            if name not in nameList:
                f.write(f"{name},{dtString}\n")
                print(f"[LOG] Attendance marked for {name}")


#Step 4: Webcam Attendance

running = False  # global flag

def start_attendance():
    global running
    running = True
    cap = cv2.VideoCapture(0)

    while running:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(img_rgb)
        encodesCurFrame = face_recognition.face_encodings(img_rgb, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            name = "UNKNOWN"
            if len(faceDis) > 0:
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex] and faceDis[matchIndex] < 0.45:
                    name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)

        cv2.imshow("Webcam", img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

#Step 5: Attendance File
    
def view_attendance():
    if not os.path.exists("Attendance.csv"):
        messagebox.showinfo("Info", "No attendance file found yet!")
        return

    top = tk.Toplevel(root)
    top.title("Attendance Sheet")

    tree = ttk.Treeview(top, columns=("Name", "Time"), show="headings")
    tree.heading("Name", text="Name")
    tree.heading("Time", text="Time")
    tree.pack(fill="both", expand=True)

    with open("Attendance.csv", "r") as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            name, time = line.strip().split(",")
            tree.insert("", "end", values=(name, time))

    tk.Button(top, text="Close", command=top.destroy,
              bg="red", fg="white", width=10).pack(pady=5)
def Restart_attendance():
    with open("Attendance.csv", "w") as f:
        f.write("Name,Time\n")
    messagebox.showinfo("Reset", "Attendance sheet has been reset!")


# Step 6: Tkinter GUI

root = tk.Tk()
root.title("Face Attendance System")
root.geometry("700x400")

tk.Button(root, text="Start Attendance", command=start_attendance,
          bg="light purple", fg="white", width=20, height=2).pack(pady=10)

tk.Button(root, text="View Attendance Sheet", command=view_attendance,
          bg="pink", fg="white", width=20, height=2).pack(pady=10)

tk.Button(root, text="Exit", command=root.destroy,
          bg="light blue", fg="white", width=20, height=2).pack(pady=10)
tk.Button(root, text="Reset", command=Restart_attendance,
          bg="teal", fg="white", width=20, height=2).pack(pady=10)


root.mainloop()
