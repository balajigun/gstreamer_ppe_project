import openvino as ov
from openvino import Core, opset8
import numpy as np


def create_test_model():
    """
    Create a very small OpenVINO model.

    Model:
        input -> multiply by 2 -> output

    This is only for learning/testing OpenVINO inference.
    It is NOT our PPE model.
    """

    input_shape = [1, 3]

    input_node = opset8.parameter(
        input_shape,
        np.float32,
        name="input"
    )

    constant_node = opset8.constant(
        np.array([2.0, 2.0, 2.0], dtype=np.float32)
    )

    multiply_node = opset8.multiply(
        input_node,
        constant_node
    )

    result_node = opset8.result(
        multiply_node
    )

    model = ov.Model(
        [result_node],
        [input_node],
        "SimpleMultiplyModel"
    )

    return model


def main():

    print("=" * 60)
    print("OPENVINO GPU INFERENCE TEST")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Create OpenVINO Core
    # -------------------------------------------------

    core = Core()

    print("\nOpenVINO initialized successfully.")

    # -------------------------------------------------
    # 2. Check available devices
    # -------------------------------------------------

    devices = core.available_devices

    print("\nAvailable devices:")

    for device in devices:
        print(f"  - {device}")

    # -------------------------------------------------
    # 3. Select GPU
    # -------------------------------------------------

    if "GPU" not in devices:

        raise RuntimeError(
            "OpenVINO GPU device is not available."
        )

    device = "GPU"

    gpu_name = core.get_property(
        device,
        "FULL_DEVICE_NAME"
    )

    print(f"\nSelected device: {device}")
    print(f"GPU: {gpu_name}")

    # -------------------------------------------------
    # 4. Create test model
    # -------------------------------------------------

    print("\nCreating test OpenVINO model...")

    model = create_test_model()

    print("Test model created successfully.")

    # -------------------------------------------------
    # 5. Compile model for GPU
    # -------------------------------------------------

    print("\nCompiling model for GPU...")

    compiled_model = core.compile_model(
        model,
        device
    )

    print("Model compiled successfully.")

    # -------------------------------------------------
    # 6. Create inference request
    # -------------------------------------------------

    infer_request = compiled_model.create_infer_request()

    print("Inference request created.")

    # -------------------------------------------------
    # 7. Prepare input
    # -------------------------------------------------

    input_data = np.array(
        [[10.0, 20.0, 30.0]],
        dtype=np.float32
    )

    print("\nInput:")
    print(input_data)

    # -------------------------------------------------
    # 8. Run inference
    # -------------------------------------------------

    print("\nRunning inference...")

    result = infer_request.infer({
        compiled_model.input(0): input_data
    })

    # -------------------------------------------------
    # 9. Get output
    # -------------------------------------------------

    output = result[
        compiled_model.output(0)
    ]

    print("\nOutput:")
    print(output)

    # -------------------------------------------------
    # 10. Verify result
    # -------------------------------------------------

    expected = input_data * 2

    if np.allclose(output, expected):

        print("\nInference result is correct.")

    else:

        print("\nWARNING: Unexpected inference result.")

    print("\n" + "=" * 60)
    print("OPENVINO GPU INFERENCE TEST SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()