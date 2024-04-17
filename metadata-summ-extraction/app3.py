import cv2
import streamlit as st
from ultralytics import YOLO
import supervision as sv

class_names = {0: "text"}

def text_detection_with_yolo(image_path, model_path):
    try:
        frame = cv2.imread(image_path)
        model = YOLO(model_path)
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = []
        cropped_images = []
        for detection in detections:
            bbox = list(detection[0])
            class_id = detection[2]
            confidence = detection[1]
            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else "Unknown"
            # label = f"{class_name} {confidence:.2f}"
            label = f""

            labels.append(label)
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_image = frame[y_min:y_max, x_min:x_max]
            cropped_images.append(cropped_image)
        annotated_image = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        return annotated_image, cropped_images
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None

def main():
    st.title("YOLO Text Detection App")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        image_path = "temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        model_path = "models/YOLOv8_text_detection_model/best.pt"
        annotated_image, bounding_boxes = text_detection_with_yolo(image_path, model_path)

        if annotated_image is not None:
            # st.image(annotated_image, caption="Processed Image", use_column_width=True)

            if bounding_boxes:
                st.subheader("Bounding boxes:")
                for box in bounding_boxes:
                    st.image(box, caption="Cropped Image", use_column_width=True)
        else:
            st.error("Error processing image.")

if __name__ == "__main__":
    main()


