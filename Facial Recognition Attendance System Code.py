import os
import cv2
import numpy as np
import face_recognition
import pandas as pd
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

# --------------------------- Configuration ---------------------------
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "face_dataset")
DATA_DIR = os.path.join(BASE_DIR, "data")
STUDENTS_CSV = os.path.join(DATA_DIR, "students.csv")

# Save attendance in Downloads (daily files)
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")

DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin"

# Ensure folders exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Ensure CSV files exist
if not os.path.exists(STUDENTS_CSV):
    with open(STUDENTS_CSV, "w", encoding="utf-8") as f:
        f.write("student_id,name,class,address,photo\n")

# --------------------------- Helpers ---------------------------
def load_student_db():
    try:
        df = pd.read_csv(STUDENTS_CSV)
    except Exception:
        df = pd.DataFrame(columns=["student_id","name","class","address","photo"])
    return df

def save_student_record(student_id, name, class_name, address, photo_filename):
    df = load_student_db()
    if str(student_id) in df['student_id'].astype(str).values:
        raise ValueError("Student ID already exists")
    new_row = pd.DataFrame([{
        "student_id": student_id,
        "name": name,
        "class": class_name,
        "address": address,
        "photo": photo_filename
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(STUDENTS_CSV, index=False)

# --------------------------- Face encodings ---------------------------
images = []
classNames = []
encodeListKnown = []

def load_images_and_names():
    global images, classNames
    images = []
    classNames = []
    files = sorted(os.listdir(DATASET_DIR))
    for fname in files:
        full = os.path.join(DATASET_DIR, fname)
        if os.path.isfile(full):
            img = cv2.imread(full)
            if img is None:
                print(f"[WARN] Could not read image: {full}")
                continue
            images.append(img)
            classNames.append(os.path.splitext(fname)[0])
    print(f"[INFO] Loaded {len(images)} images for encoding.")

def findEncodings():
    global encodeListKnown
    encodeListKnown = []
    for idx, img in enumerate(images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(img_rgb)
        if encs:
            encodeListKnown.append(encs[0])
            print(f"[OK] Encoded {classNames[idx]}")
        else:
            print(f"[WARN] No face found in {classNames[idx]} — skipping")
    print(f"[INFO] Total encodings: {len(encodeListKnown)}")

def reload_encodings():
    load_images_and_names()
    findEncodings()

reload_encodings()

# --------------------------- Attendance logging ---------------------------
def markAttendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # File for today in Downloads
    filename = os.path.join(DOWNLOADS, f"Attendance_{date_str}.csv")

    # If file doesn't exist, create with header
    if not os.path.isfile(filename):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    # Read existing entries
    with open(filename, "r", encoding="utf-8") as f:
        existing = [line.split(",")[0] for line in f.readlines()[1:]]

    # Only mark once per day
    if name not in existing:
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])
        print(f"[LOG] Attendance marked for {name} at {time_str}")

# --------------------------- GUI App ---------------------------
class FaceAttendanceApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Attendance System - Login")
        master.geometry("420x240")
        self.create_login_ui()

    def create_login_ui(self):
        for widget in self.master.winfo_children():
            widget.destroy()
        tk.Label(self.master, text="Username").pack(pady=(20,5))
        self.username_entry = tk.Entry(self.master)
        self.username_entry.pack()
        tk.Label(self.master, text="Password").pack(pady=(10,5))
        self.password_entry = tk.Entry(self.master, show="*")
        self.password_entry.pack()
        tk.Button(self.master, text="Login", width=12, command=self.check_login, bg = "teal", fg = "white").pack(pady=12)
        tk.Label(self.master, text=f"(default: {DEFAULT_USERNAME} / {DEFAULT_PASSWORD})", fg="light blue").pack()

    def check_login(self):
        user = self.username_entry.get().strip()
        pwd = self.password_entry.get().strip()
        if user == DEFAULT_USERNAME and pwd == DEFAULT_PASSWORD:
            self.create_main_ui()
        else:
            messagebox.showerror("Login failed", "Incorrect username or password")

    def create_main_ui(self):
        self.master.title("Face Attendance System - Main")
        for widget in self.master.winfo_children():
            widget.destroy()
        tk.Button(self.master, text="Register Student", width=24, height=2, command=self.open_register_window, bg = "light pink", fg = "white").pack(pady=8)
        tk.Button(self.master, text="Start Attendance", width=24, height=2, command=self.start_attendance, bg = "purple", fg = "white").pack(pady=8)
        tk.Button(self.master, text="View Attendance Sheet", width=24, height=2, command=self.view_attendance, bg = "indigo", fg = "white").pack(pady=8)
        tk.Button(self.master, text="Reload Encodings", width=24, height=2, command=self.reload_encodings_ui, bg = "light blue", fg = "white").pack(pady=8)
        tk.Button(self.master, text="Exit", width=24, height=2, command=self.master.destroy, bg = "teal", fg = "white").pack(pady=8)

    def open_register_window(self):
        reg = tk.Toplevel(self.master)
        reg.title("Register Student")
        reg.geometry("420x460")
        tk.Label(reg, text="Student ID").pack(pady=(10,0))
        sid_entry = tk.Entry(reg)
        sid_entry.pack()
        tk.Label(reg, text="Name").pack(pady=(10,0))
        name_entry = tk.Entry(reg)
        name_entry.pack()
        tk.Label(reg, text="Class").pack(pady=(10,0))
        class_entry = tk.Entry(reg)
        class_entry.pack()
        tk.Label(reg, text="Address").pack(pady=(10,0))
        addr_entry = tk.Entry(reg)
        addr_entry.pack()
        info_label = tk.Label(reg, text="Capture a frontal photo using webcam. Filename will be studentID_name.jpg")
        info_label.pack(pady=10)
        capture_btn = tk.Button(reg, text="Capture Photo", width=20, command=lambda: self.capture_photo_for_registration(sid_entry, name_entry, class_entry, addr_entry, reg))
        capture_btn.pack(pady=5)
        tk.Button(reg, text="Close", width=12, command=reg.destroy).pack(pady=10)

    def capture_photo_for_registration(self, sid_entry, name_entry, class_entry, addr_entry, parent_window):
        sid = sid_entry.get().strip()
        name = name_entry.get().strip()
        class_name = class_entry.get().strip()
        address = addr_entry.get().strip()
        if not sid or not name:
            messagebox.showerror("Missing data", "Please enter at least Student ID and Name")
            return
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Camera error", "Could not access the webcam")
            return
        messagebox.showinfo("Instruction", "Position the student's face in front of the camera and press 'c' to capture. Press 'q' to cancel.")
        captured = False
        filepath = None
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            cv2.imshow("Capture Photo - Press c to capture, q to cancel", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('c'):
                safe_name = f"{sid}_{name}".replace(" ", "_")
                filename = f"{safe_name}.jpg"
                filepath = os.path.join(DATASET_DIR, filename)
                cv2.imwrite(filepath, frame)
                captured = True
                break
            elif k == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()
        if captured:
            try:
                save_student_record(sid, name, class_name, address, filename)
            except Exception as e:
                try:
                    if filepath and os.path.exists(filepath):
                        os.remove(filepath)
                except Exception:
                    pass
                messagebox.showerror("Error", f"Could not save student record: {e}")
                return
            reload_encodings()
            messagebox.showinfo("Success", f"Student {name} registered and photo saved as {filename}")
            parent_window.destroy()

    def start_attendance(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera error", "Could not access the webcam")
            return
        df_students = load_student_db()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(img_rgb)
            encodesCurFrame = face_recognition.face_encodings(img_rgb, facesCurFrame)
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                name = "UNKNOWN"
                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex] and faceDis[matchIndex] < 0.48:
                        name = classNames[matchIndex]
                y1, x2, y2, x1 = [v * 4 for v in faceLoc]
                color = (0,255,0) if name != "UNKNOWN" else (0,0,255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                display_text = ""
                if name != "UNKNOWN":
                    df = df_students
                    matched = df[df['photo'].str.upper() == (name + ".JPG").upper()]
                    if matched.empty:
                        matched = df[df['photo'].str.upper() == (name + ".PNG").upper()]
                    if not matched.empty:
                        row = matched.iloc[0]
                        display_text = f"{row['student_id']} | {row['name']}"
                        markAttendance(row['name'])
                    else:
                        display_text = "Data not found"
                        markAttendance("UNKNOWN")
                else:
                    display_text = "Data not found"
                    markAttendance("UNKNOWN")
                cv2.putText(frame, name.upper(), (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                y_text = y2 + 25
                for i, chunk in enumerate(split_text(display_text, 50)):
                    cv2.putText(frame, chunk, (x1 + 6, y_text + i*20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow("Attendance - Press q to quit", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def view_attendance(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(DOWNLOADS, f"Attendance_{date_str}.csv")

        if not os.path.exists(filename):
            messagebox.showinfo("Info", f"No attendance file found for {date_str}!")
            return

        top = tk.Toplevel(self.master)
        top.title(f"Attendance Sheet - {date_str}")
        top.geometry("700x450")

        tree = ttk.Treeview(top, columns=("Name", "Date", "Time"), show="headings")
        tree.heading("Name", text="Name")
        tree.heading("Date", text="Date")
        tree.heading("Time", text="Time")
        tree.pack(fill="both", expand=True)

        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    tree.insert("", "end", values=(parts[0], parts[1], parts[2]))

        tk.Button(top, text="Close", command=top.destroy).pack(pady=5)

    def reload_encodings_ui(self):
        reload_encodings()
        messagebox.showinfo("Reload", "Encodings reloaded from dataset folder.")

# --------------------------- Utilities ---------------------------
def split_text(text, width):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if not lines:
        return [""]
    return lines

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()
