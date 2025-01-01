from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_data, model_type='cnn'):
    """
    Evaluate model performance
    """
    if model_type == 'cnn':
        predictions = model.predict(test_data)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.concatenate([y for _, y in test_data])
    else:  # YOLO
        results = model.val()
        # Extract metrics from YOLO results
        return results
    
    # Generate classification report
    report = classification_report(y_true, y_pred, 
                                 target_names=['Healthy', 'Bleached'])
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return report, cm