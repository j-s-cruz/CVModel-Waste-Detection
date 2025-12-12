from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

# Load YOLOv8 pretrained model
model = YOLO("waste_sorter.pt")

# Set paths
input_folder = "test_images/"      # folder of your waste image dataset
output_folder = "results/"
os.makedirs(output_folder, exist_ok=True)

# Run inference on all images
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.jpg', '.png'))]

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)

    # Perform inference
    results = model(img_path)

    # Save annotated prediction image
    results[0].save(os.path.join(output_folder, img_name))

    print(f"Processed: {img_name}")

# Compute evaluation metrics
if os.path.exists("labels"):
    metrics = model.val(data="data.yaml")   # requires data.yaml for test dataset
    print(metrics)

# Function to plot metrics
def plot_metrics(metrics):
    classes = metrics['metrics/class']
    x = range(len(classes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar([i - width for i in x], metrics['metrics/precision_cls'], width, label="Precision")
    ax.bar(x, metrics['metrics/recall_cls'], width, label="Recall")
    ax.bar([i + width for i in x], metrics['metrics/F1_cls'], width, label="F1 Score")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("YOLOv8 Evaluation Metrics per Class")
    ax.legend()

    plt.show()
plot_metrics(metrics)
