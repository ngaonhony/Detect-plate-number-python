import os

# Lệnh huấn luyện mô hình YOLOv5
os.system('python C:/Users/ADMIN/PycharmProjects/Detect-plate-number/yolov5/train.py --img 640 --batch 16 --epochs 1 --data data.yaml --weights yolov5s.pt')