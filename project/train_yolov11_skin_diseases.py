import os
import cv2
from ultralytics import YOLO

def train_yolov11_model():
    """
    Trains a YOLOv11 model on the skin diseases dataset.
    """
    # Define the path to your data.yaml file
    # Ensure this path is correct relative to where you run the script,
    # or provide an absolute path.
    data_yaml_path = 'data.yaml'

    # Check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found.")
        print("Please ensure 'data.yaml' is in the same directory as this script,")
        print("or update the 'data_yaml_path' variable to its correct location.")
        return

    # 1. Load a pre-trained YOLOv11 model
    # You can choose different model sizes: 'yolov11n.pt' (nano), 'yolov11s.pt' (small),
    # 'yolov11m.pt' (medium), 'yolov11l.pt' (large), 'yolov11x.pt' (extra large).
    # 'yolov11n.pt' is a good starting point for quick training and smaller models.
    print("Loading YOLOv11n model...")
    model = YOLO('yolov11n.pt')  # Loads a pre-trained nano model

    # 2. Train the model
    # The 'data' argument points to your data.yaml file.
    # 'epochs': Number of training epochs. Start with a smaller number (e.g., 50-100)
    #           and increase if the model is still improving.
    # 'imgsz': Image size for training. Your dataset is resized to 640x640.
    # 'batch': Batch size. Adjust based on your GPU memory. Common values: 16, 32, 64.
    #          Reduce if you get out-of-memory errors.
    # 'name': Name for the training run, results will be saved in runs/detect/name.
    print("Starting model training...")
    results = model.train(
        data=data_yaml_path,
        epochs=100,  # You can adjust this number
        imgsz=640,
        batch=16,    # Adjust based on your GPU memory
        name='skin_diseases_yolov11_run'
    )

    print("\nTraining complete!")
    print("Results are saved in the 'runs/detect/skin_diseases_yolov11_run' directory.")
    print("You can find the best model weights (best.pt) and other metrics there.")

    # 3. Evaluate the model on the validation set
    print("\nEvaluating the trained model on the validation set...")
    metrics = model.val()  # No arguments needed if data.yaml is configured correctly

    print("\nValidation Metrics:")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")

    # 4. (Optional) Predict on new images or the test set
    # To predict on the test set specified in data.yaml:
    # print("\nRunning prediction on the test set...")
    # model.predict(source=os.path.join(os.path.dirname(data_yaml_path), 'test/images'), save=True, conf=0.25)
    # print("Predictions on test set saved in 'runs/detect/predict' directory.")

if __name__ == "__main__":
    train_yolov11_model()

