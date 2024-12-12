# Cài đặt các thư viện cần thiết
# pip install easyocr
# pip install opencv-python==4.5.4.60
# pip install opencv-contrib-python==4.5.4.60
# pip install matplotlib
# pip install pillow
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, filedialog, messagebox
import torch
from easyocr import Reader
import os

class LicensePlateRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition")
        self.root.geometry("800x600")
        self.root.configure(bg="white")

        self.label = Label(root, text="License Plate Recognition", font=("Arial", 18), bg="white")
        self.label.pack(pady=20)

        self.image_label = Label(root)
        self.image_label.pack()

        self.result_label = Label(root, text="", font=("Arial", 14), bg="white", fg="green")
        self.result_label.pack(pady=20)

        self.select_button = Button(root, text="Select Image", command=self.load_image, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.select_button.pack(pady=10)

        self.recognize_button = Button(root, text="Recognize License Plate", command=self.recognize_plate, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.recognize_button.pack(pady=10)

        self.image_path = None
        self.result_text = ""

        # Tải mô hình YOLO từ PyTorch Hub
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

    def load_image(self):
        # Chọn một file hình ảnh
        self.image_path = filedialog.askopenfilename(title="Select Image", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
        if not self.image_path:
            return

        # Hiển thị hình ảnh đã chọn
        self.display_image(self.image_path)

    def display_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((400, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def recognize_plate(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        img = cv2.imread(self.image_path)
        img_resized = cv2.resize(img, (800, 600))

        # Sử dụng YOLO để phát hiện biển số
        results = self.model(img_resized)
        detections = results.pandas().xyxy[0]

        number_plate_shape = None

        for index, row in detections.iterrows():
            if row['confidence'] > 0.4:  # Ngưỡng độ tin cậy
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                number_plate_shape = (x1, y1, x2, y2)
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Vẽ hình chữ nhật xung quanh biển số
                break

        if number_plate_shape is not None:
            x1, y1, x2, y2 = number_plate_shape
            number_plate = img_resized[y1:y2, x1:x2]

            # Tiền xử lý ảnh biển số
            number_plate_gray = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)
            number_plate_enhanced = cv2.equalizeHist(number_plate_gray)

            # Nhận diện văn bản
            reader = Reader(['en', 'vi'])  # Thêm ngôn ngữ Việt
            detection = reader.readtext(number_plate_enhanced)

            if len(detection) == 0:
                self.result_text = "Không tìm thấy bảng số xe"
            else:
                self.result_text = f"Biển số: {detection[0][1]}"
        else:
            self.result_text = "Không tìm thấy biển số"

        # Hiển thị kết quả
        self.result_label.config(text=self.result_text)

        # Ghi kết quả vào file
        with open("license_plate_result.txt", "a", encoding="utf-8") as file:
            file.write(self.result_text + "\n")

        messagebox.showinfo("Result Saved", "The result has been saved to license_plate_result.txt")

# Tạo cửa sổ chính
root = Tk()
app = LicensePlateRecognizer(root)
root.mainloop()