"""
YOLO + OpenVINO GPU Inference Test

Purpose:
    Perform direct OpenVINO inference using the exported YOLO model
    on Intel Iris Xe GPU.

Pipeline:

    Image
      |
      v
    OpenCV
      |
      v
    Preprocessing
      |
      v
    OpenVINO
      |
      v
    Intel GPU
      |
      v
    YOLO output
      |
      v
    Post-processing
      |
      v
    NMS
      |
      v
    Annotated image
"""

from pathlib import Path

import cv2
import numpy as np
import openvino as ov


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path(
    r"D:\gstreamer_ppe_project\models\best_openvino_model"
)

IMAGE_PATH = Path(
    r"D:\ppe_monitoring\data\tracking_evaluation\frames\frame_0000.jpg"
)

OUTPUT_PATH = Path(
    r"D:\gstreamer_ppe_project\output\openvino_gpu_result.jpg"
)

DEVICE = "GPU"

INPUT_SIZE = 640

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}


# ============================================================
# FIND OPENVINO MODEL
# ============================================================

def find_model(model_dir: Path) -> Path:

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist:\n{model_dir}"
        )

    xml_files = list(model_dir.glob("*.xml"))

    if not xml_files:
        raise FileNotFoundError(
            f"No OpenVINO .xml model found in:\n{model_dir}"
        )

    return xml_files[0]


# ============================================================
# LETTERBOX
# ============================================================

def letterbox(
    image,
    new_shape=(640, 640),
    color=(114, 114, 114),
):
    """
    Resize image while maintaining aspect ratio.

    Returns:
        resized image
        scale ratio
        padding (left, top)
    """

    original_height, original_width = image.shape[:2]

    new_height, new_width = new_shape

    ratio = min(
        new_width / original_width,
        new_height / original_height,
    )

    resized_width = int(round(original_width * ratio))
    resized_height = int(round(original_height * ratio))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_width = new_width - resized_width
    pad_height = new_height - resized_height

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    resized = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return resized, ratio, (pad_left, pad_top)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess(image):

    image_resized, ratio, padding = letterbox(
        image,
        (INPUT_SIZE, INPUT_SIZE),
    )

    # BGR -> RGB
    image_rgb = cv2.cvtColor(
        image_resized,
        cv2.COLOR_BGR2RGB,
    )

    # uint8 -> float32
    image_rgb = image_rgb.astype(
        np.float32
    )

    # Normalize
    image_rgb /= 255.0

    # HWC -> CHW
    image_chw = np.transpose(
        image_rgb,
        (2, 0, 1),
    )

    # Add batch dimension
    input_tensor = np.expand_dims(
        image_chw,
        axis=0,
    )

    return input_tensor, ratio, padding


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1,
    )

    intersection_height = max(
        0,
        y2 - y1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# NMS
# ============================================================

def nms(boxes, scores, iou_threshold):

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:

        current = order[0]

        keep.append(current)

        remaining = []

        for index in order[1:]:

            iou = calculate_iou(
                boxes[current],
                boxes[index],
            )

            if iou < iou_threshold:
                remaining.append(index)

        order = np.array(
            remaining,
            dtype=np.int32,
        )

    return keep


# ============================================================
# POSTPROCESS YOLO OUTPUT
# ============================================================

def postprocess(
    output,
    original_image,
    ratio,
    padding,
):
    """
    YOLO exported output:

        (1, 14, 8400)

    14 values:

        4 box coordinates
        10 class scores

    Output format:

        [x, y, w, h, class0, class1, ...]
    """

    original_height, original_width = (
        original_image.shape[:2]
    )

    # Remove batch dimension
    predictions = output[0]

    # YOLO output is:
    #
    #   (14, 8400)
    #
    # Convert to:
    #
    #   (8400, 14)

    predictions = predictions.transpose(1, 0)

    boxes = []
    scores = []
    class_ids = []

    # --------------------------------------------------------
    # Decode detections
    # --------------------------------------------------------

    for prediction in predictions:

        x_center = prediction[0]
        y_center = prediction[1]

        width = prediction[2]
        height = prediction[3]

        class_scores = prediction[4:]

        class_id = int(
            np.argmax(class_scores)
        )

        confidence = float(
            class_scores[class_id]
        )

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Convert xywh -> xyxy
        x1 = x_center - width / 2
        y1 = y_center - height / 2

        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Remove letterbox padding
        pad_x, pad_y = padding

        x1 -= pad_x
        x2 -= pad_x

        y1 -= pad_y
        y2 -= pad_y

        # Undo scaling
        x1 /= ratio
        x2 /= ratio

        y1 /= ratio
        y2 /= ratio

        # Clip to original image
        x1 = max(
            0,
            min(x1, original_width - 1),
        )

        y1 = max(
            0,
            min(y1, original_height - 1),
        )

        x2 = max(
            0,
            min(x2, original_width - 1),
        )

        y2 = max(
            0,
            min(y2, original_height - 1),
        )

        boxes.append(
            [x1, y1, x2, y2]
        )

        scores.append(confidence)

        class_ids.append(class_id)

    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    keep = nms(
        boxes,
        scores,
        IOU_THRESHOLD,
    )

    detections = []

    for index in keep:

        detections.append(
            {
                "box": boxes[index],
                "confidence": scores[index],
                "class_id": class_ids[index],
            }
        )

    return detections


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image,
    detections,
):

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        confidence = detection["confidence"]

        class_id = detection["class_id"]

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}",
        )

        x1 = int(x1)
        y1 = int(y1)

        x2 = int(x2)
        y2 = int(y2)

        # Bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        # Label background
        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )
        )

        cv2.rectangle(
            image,
            (
                x1,
                max(
                    0,
                    y1 - text_height - baseline,
                ),
            ),
            (
                x1 + text_width,
                y1,
            ),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            image,
            label,
            (
                x1,
                max(
                    text_height,
                    y1 - baseline,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLO + OPENVINO GPU INFERENCE TEST")
    print("=" * 70)

    print()
    print(f"Model directory : {MODEL_DIR}")
    print(f"Image           : {IMAGE_PATH}")
    print(f"Device          : {DEVICE}")
    print()

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            f"Image does not exist:\n{IMAGE_PATH}"
        )

    # --------------------------------------------------------
    # Find model
    # --------------------------------------------------------

    model_path = find_model(
        MODEL_DIR
    )

    print(
        f"OpenVINO model : {model_path}"
    )

    # --------------------------------------------------------
    # OpenVINO Core
    # --------------------------------------------------------

    print()
    print("Initializing OpenVINO...")

    core = ov.Core()

    print("OpenVINO initialized successfully.")

    # --------------------------------------------------------
    # Check devices
    # --------------------------------------------------------

    print()
    print("Available OpenVINO devices:")

    for device in core.available_devices:

        print(
            f"  - {device}"
        )

    if DEVICE not in core.available_devices:

        raise RuntimeError(
            f"\nOpenVINO device '{DEVICE}' "
            f"is not available."
        )

    # --------------------------------------------------------
    # GPU information
    # --------------------------------------------------------

    print()

    try:

        gpu_name = core.get_property(
            DEVICE,
            "FULL_DEVICE_NAME",
        )

        print(
            f"GPU : {gpu_name}"
        )

    except Exception:

        print(
            "GPU device information unavailable."
        )

    # --------------------------------------------------------
    # Read model
    # --------------------------------------------------------

    print()
    print("Reading OpenVINO model...")

    model = core.read_model(
        model_path
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Model input information
    # --------------------------------------------------------

    print()
    print("Model input:")

    for input_layer in model.inputs:

        print(
            f"  Shape : {input_layer.shape}"
        )

        print(
            f"  Type  : {input_layer.element_type}"
        )

    # --------------------------------------------------------
    # Model output information
    # --------------------------------------------------------

    print()
    print("Model output:")

    for output_layer in model.outputs:

        print(
            f"  Shape : {output_layer.shape}"
        )

        print(
            f"  Type  : {output_layer.element_type}"
        )

    # --------------------------------------------------------
    # Compile model on GPU
    # --------------------------------------------------------

    print()
    print(
        "Compiling model for Intel GPU..."
    )

    compiled_model = core.compile_model(
        model,
        DEVICE,
    )

    print(
        "Model compiled successfully on GPU."
    )

    # --------------------------------------------------------
    # Create inference request
    # --------------------------------------------------------

    infer_request = (
        compiled_model.create_infer_request()
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    print()
    print("Loading test image...")

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:

        raise RuntimeError(
            f"OpenCV failed to load image:\n"
            f"{IMAGE_PATH}"
        )

    print(
        f"Image size : "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print()
    print("Preprocessing image...")

    input_tensor, ratio, padding = (
        preprocess(image)
    )

    print(
        f"Input tensor shape : "
        f"{input_tensor.shape}"
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    print()
    print("Running inference on Intel GPU...")

    results = infer_request.infer(
        {0: input_tensor}
    )

    print(
        "GPU inference completed successfully."
    )

    # --------------------------------------------------------
    # Retrieve output
    # --------------------------------------------------------

    output_tensor = next(
        iter(results.values())
    )

    print()
    print(
        f"Raw output shape : "
        f"{output_tensor.shape}"
    )

    # --------------------------------------------------------
    # Postprocess
    # --------------------------------------------------------

    print()
    print("Post-processing YOLO output...")

    detections = postprocess(
        output_tensor,
        image,
        ratio,
        padding,
    )

    print(
        f"Detections after NMS : "
        f"{len(detections)}"
    )

    # --------------------------------------------------------
    # Print detections
    # --------------------------------------------------------

    print()
    print("Detections:")

    if len(detections) == 0:

        print("  No objects detected.")

    else:

        for i, detection in enumerate(
            detections,
            start=1,
        ):

            class_id = detection[
                "class_id"
            ]

            class_name = CLASS_NAMES.get(
                class_id,
                f"class_{class_id}",
            )

            confidence = detection[
                "confidence"
            ]

            box = detection["box"]

            print(
                f"  {i}. "
                f"{class_name:<18} "
                f"confidence={confidence:.3f} "
                f"box={box}"
            )

    # --------------------------------------------------------
    # Draw detections
    # --------------------------------------------------------

    annotated_image = draw_detections(
        image.copy(),
        detections,
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(OUTPUT_PATH),
        annotated_image,
    )

    if not success:

        raise RuntimeError(
            f"Failed to save output image:\n"
            f"{OUTPUT_PATH}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INFERENCE SUCCESSFUL")
    print("=" * 70)

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Input image  : {IMAGE_PATH}"
    )

    print(
        f"Output image : {OUTPUT_PATH}"
    )

    print(
        f"Detections   : {len(detections)}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()