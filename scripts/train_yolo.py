from ultralytics import YOLO

def train_yolo(data_yaml_path):
    """
    Train YOLO model
    """
    model = YOLO('yolo11n.pt')
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        save=True
    )
    return model, results
