import cv2
import numpy as np

def detect_and_crop_coral(image_path, model):
    """
    Detect and crop coral regions from images
    """
    image = cv2.imread(image_path)
    results = model(image)
    
    cropped_images = []
    for det in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, det[:4])
        crop = image[y1:y2, x1:x2]
        cropped_images.append(crop)
    
    return cropped_images

# combined_model.py
class CombinedCoralClassifier:
    def __init__(self, cnn_model, yolo_model):
        self.cnn_model = cnn_model
        self.yolo_model = yolo_model
    
    def predict(self, image_path):
        """
        Combine YOLO detection and CNN classification
        """
        # First detect corals using YOLO
        crops = detect_and_crop_coral(image_path, self.yolo_model)
        
        predictions = []
        for crop in crops:
            # Preprocess for CNN
            crop = cv2.resize(crop, (224, 224))
            crop = crop / 255.0
            crop = np.expand_dims(crop, axis=0)
            
            # Get CNN prediction
            pred = self.cnn_model.predict(crop)
            class_idx = np.argmax(pred)
            confidence = np.max(pred)
            
            predictions.append({
                'class': 'Healthy' if class_idx == 0 else 'Bleached',
                'confidence': float(confidence)
            })
        
        return predictions