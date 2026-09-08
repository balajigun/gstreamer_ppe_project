from openvino import Core


def main():

    print("=" * 60)
    print("OPENVINO DEVICE TEST")
    print("=" * 60)

    # -------------------------------------------------
    # Create OpenVINO Core
    # -------------------------------------------------

    core = Core()

    print("\nOpenVINO initialized successfully.")

    # -------------------------------------------------
    # List available devices
    # -------------------------------------------------

    devices = core.available_devices

    print("\nAvailable OpenVINO devices:")

    for device in devices:

        print(f"  - {device}")

    # -------------------------------------------------
    # Check GPU
    # -------------------------------------------------

    if "GPU" not in devices:

        print(
            "\nWARNING: OpenVINO GPU device is not available."
        )

        print(
            "The system will need the Intel GPU plugin/runtime "
            "before GPU inference can be used."
        )

        return

    # -------------------------------------------------
    # Get GPU information
    # -------------------------------------------------

    gpu_name = core.get_property(
        "GPU",
        "FULL_DEVICE_NAME"
    )

    print(
        f"\nGPU detected: {gpu_name}"
    )

    # -------------------------------------------------
    # Get GPU driver information
    # -------------------------------------------------

    try:

        driver_version = core.get_property(
            "GPU",
            "DEVICE_ID"
        )

        print(
            f"GPU device ID: {driver_version}"
        )

    except Exception:

        pass

    print("\nOpenVINO GPU device test successful.")


if __name__ == "__main__":
    main()