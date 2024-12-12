import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, filedialog, messagebox, Canvas, Frame
from easyocr import Reader

class LicensePlateRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition")
        self.root.geometry("800x600")
        self.root.configure(bg="white")

        # Create a frame with a border
        self.border_frame = Frame(root, bd=5, relief="groove", bg="white")
        self.border_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.label = Label(self.border_frame, text="License Plate Recognition", font=("Arial Black", 18), bg="white")
        self.label.grid(row=0, column=0, columnspan=2, pady=20)

        # Add a dividing line
        self.divider_canvas = Canvas(self.border_frame, height=2, bg="black", bd=0, highlightthickness=0)
        self.divider_canvas.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky="ew")

        # Make the canvas stretch across the frame
        self.border_frame.grid_columnconfigure(0, weight=1)
        self.border_frame.grid_columnconfigure(1, weight=2)  # Allow more space for the image

        # Frame chứa các nút
        self.button_frame = Frame(self.border_frame, bg="white")
        self.button_frame.grid(row=2, column=0, padx=30, pady=10, sticky="ns")

        # Create buttons with equal width
        self.select_button = Button(self.button_frame, text="Select Image", command=self.select_image, font=("Arial Black", 12), bg="#4CAF50", fg="white", width=20)
        self.select_button.grid(row=0, column=0, padx=5, pady=10, sticky="ew")

        self.recognize_button = Button(self.button_frame, text="Recognize License Plate", command=self.recognize_plate, font=("Arial Black", 12), bg="#4CAF50", fg="white", width=20)
        self.recognize_button.grid(row=1, column=0, padx=5, pady=100, sticky="ew")

        # Make the buttons expand equally
        self.button_frame.grid_columnconfigure(0, weight=1)

        # Tạo khung ảnh với Canvas
        self.image_frame = Frame(self.border_frame, bd=2, relief="solid", bg="white")
        self.image_frame.grid(row=2, column=1, padx=30, pady=10, sticky="nsew")  # Position on the right

        self.canvas = Canvas(self.image_frame, width=400, height=300, bg="white")
        self.canvas.pack()

        # Vẽ viền đen cho khung ảnh
        self.canvas.create_rectangle(1, 1, 399, 299, outline="white", width=1)

        self.result_label = Label(self.border_frame, text="", font=("Time news roman", 14), bg="white", fg="red")
        self.result_label.grid(row=3, column=0, columnspan=2, pady=20)

        self.image_path = None
        self.result_text = ""

    def select_image(self):

        self.image_path = filedialog.askopenfilename(
            initialdir="./images", title="Select Image"
            ,filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))

        if not self.image_path:
            return

        # Hiển thị ảnh đã chọn
        img = Image.open(self.image_path)
        img = img.resize((400, 300))
        img_tk = ImageTk.PhotoImage(img)

        # Xóa canvas trước khi vẽ ảnh mới
        self.canvas.delete("all")
        self.canvas.create_rectangle(1, 1, 399, 299, outline="black", width=1)  # Vẽ lại viền

        self.canvas.create_image(200, 150, image=img_tk)
        self.canvas.image = img_tk  # Giữ tham chiếu đến ảnh

    def recognize_plate(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        # Nhận diện biển số xe
        img = cv2.imread(self.image_path)
        img = cv2.resize(img, (800, 600))

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
            reader = Reader(['en'])
            detection = reader.readtext(number_plate)
            if len(detection) == 0:
                self.result_text = "Không thấy bảng số xe"
            else:
                self.result_text = f"Biển số: {detection[0][1]}"
                cv2.drawContours(img,[number_plate_shape], -1, (255, 0, 0), 3)
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