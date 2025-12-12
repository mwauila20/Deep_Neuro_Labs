from ultralytics import YOLO

# Загрузка предобученной модели YOLOv5
model = YOLO("yolov8s.pt")

# Путь к YAML файлу, который содержит данные для обучения
data_yaml = "data1.yaml"

# Обучение модели
model.train(data=data_yaml, epochs=50, batch=16, imgsz=640)

# Сохранение обученной модели
model.save("trained_model.pt")

# Тестирование модели на новом изображении
#results = model("path/to/test/image.jpg")

# Показ результатов
#results.show()