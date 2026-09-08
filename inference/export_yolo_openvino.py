from ultralytics import YOLO


MODEL_PATH = r"D:\gstreamer_ppe_project\models\best.pt"


def main():

    print("=" * 60)
    print("YOLO → OPENVINO EXPORT")
    print("=" * 60)

    print(f"\nLoading model:")
    print(MODEL_PATH)

    model = YOLO(MODEL_PATH)

    print("\nModel loaded successfully.")

    print("\nExporting model to OpenVINO...")

    export_path = model.export(
        format="openvino"
    )

    print("\nOpenVINO export completed successfully.")

    print(f"\nExported model:")
    print(export_path)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()