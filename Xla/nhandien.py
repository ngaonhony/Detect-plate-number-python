import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, filedialog, messagebox, Canvas, Frame
from easyocr import Reader
import torch

class LicensePlateRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        self.border_frame = Frame(root, bd=5, relief="groove", bg="white")
        self.border_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.label = Label(self.border_frame, text="License Plate Recognition", font=("Arial Black", 18), bg="white")
        self.label.grid(row=0, column=0, columnspan=2, pady=20)

        self.divider_canvas = Canvas(self.border_frame, height=2, bg="black", bd=0, highlightthickness=0)
        self.divider_canvas.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky="ew")

        self.button_frame = Frame(self.border_frame, bg="white")
        self.button_frame.grid(row=2, column=0, padx=30, pady=10, sticky="ns")

        self.select_button = Button(self.button_frame, text="Select Image", command=self.select_image, font=("Arial Black", 12), bg="#4CAF50", fg="white", width=20)
        self.select_button.grid(row=0, column=0, padx=5, pady=10, sticky="ew")

        self.recognize_button = Button(self.button_frame, text="Recognize License Plate", command=self.recognize_plate, font=("Arial Black", 12), bg="#4CAF50", fg="white", width=20)
        self.recognize_button.grid(row=1, column=0, padx=5, pady=100, sticky="ew")

        self.image_frame = Frame(self.border_frame, bd=5, relief="groove", bg="#ffffff")
        self.image_frame.grid(row=2, column=1, padx=30, pady=10, sticky="nsew")

        self.canvas = Canvas(self.image_frame, width=400, height=300, bg="#f9f9f9")
        self.canvas.pack()

        self.cropped_frame = Frame(self.border_frame, bd=5, relief="groove", bg="#ffffff")
        self.cropped_frame.grid(row=3, column=1, padx=30, pady=10, sticky="nsew")

        # Tạo canvas với nền trong suốt
        self.cropped_canvas = Canvas(self.cropped_frame, width=400, height=100, bg="#f9f9f9", highlightthickness=0)
        self.cropped_canvas.pack()

        # Kết quả nằm trong cropped_canvas
        self.result_text_label = Label(self.cropped_frame, text="", font=("Time News Roman", 14), bg="#f9f9f9", fg="red")
        self.result_text_label.place(relx=0.5, rely=0.5, anchor="center")  # Đặt nhãn giữa canvas

        self.image_path = None
        self.result_text = ""
        self.cropped_image = None

        # Load YOLO model
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(initialdir="./images", title="Select Image",
                                                     filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
        if not self.image_path:
            return

        img = Image.open(self.image_path)
        img = img.resize((400, 300))
        img_tk = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_rectangle(1, 1, 399, 299, outline="black", width=1)

        self.canvas.create_image(200, 150, image=img_tk)
        self.canvas.image = img_tk  # Giữ tham chiếu đến ảnh

    def recognize_plate(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        img = cv2.imread(self.image_path)
        img_resized = cv2.resize(img, (800, 600))

        # Try EasyOCR first
        self.result_text = self.easyocr_recognize(img_resized)

        if not self.result_text:  # Nếu EasyOCR không thành công, thử YOLO
            self.result_text = self.yolo_recognize(img_resized)

        # Cập nhật giao diện với kết quả
        self.result_text_label.config(text=self.result_text)  # Cập nhật nhãn kết quả

        # Ghi kết quả vào file
        with open("license_plate_result.txt", "a", encoding="utf-8") as file:
            file.write(self.result_text + "\n")

        messagebox.showinfo("Result Saved", "The result has been saved to license_plate_result.txt")

    def easyocr_recognize(self, img):
        reader = Reader(['en'])
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        edged = cv2.Canny(blurred, 10, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        number_plate_shape = None

        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approximation) == 4:
                number_plate_shape = approximation
                break

        if number_plate_shape is not None:
            (x, y, w, h) = cv2.boundingRect(number_plate_shape)
            number_plate = grayscale[y:y + h, x:x + w]
            detection = reader.readtext(number_plate)

            if len(detection) == 0:
                return ""
            else:
                return f"Biển số (EasyOCR): {detection[0][1]}"
        else:
            return ""

    def yolo_recognize(self, img):
        results = self.yolo_model(img)
        detections = results.pandas().xyxy[0]

        for index, row in detections.iterrows():
            if row['confidence'] > 0.4:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                number_plate = img[y1:y2, x1:x2]
                number_plate_gray = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)
                detection = Reader(['en', 'vi']).readtext(number_plate_gray)

                if len(detection) == 0:
                    return "Không tìm thấy bảng số xe"
                else:
                    return f"Biển số (YOLO): {detection[0][1]}"

        return "Không tìm thấy biển số"

# Tạo cửa sổ chính
root = Tk()
app = LicensePlateRecognizer(root)
root.mainloop()