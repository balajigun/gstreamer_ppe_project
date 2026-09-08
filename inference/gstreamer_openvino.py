import sys
import time

import gi
import numpy as np
import cv2
import openvino as ov

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = r"D:\ppe_monitoring\data\input_videos\factory.mp4"

MODEL_PATH = (
    r"D:\gstreamer_ppe_project\models"
    r"\best_openvino_model\best.xml"
)

DEVICE = "GPU"

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

CONFIDENCE_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45


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
# GSTREAMER + OPENVINO PIPELINE
# ============================================================

class GStreamerOpenVINO:

    def __init__(self):

        print("=" * 70)
        print("GSTREAMER + OPENVINO GPU PPE INFERENCE")
        print("=" * 70)

        print(f"Video : {VIDEO_PATH}")
        print(f"Model : {MODEL_PATH}")
        print(f"Device: {DEVICE}")
        print("=" * 70)

        # ----------------------------------------------------
        # Initialize GStreamer
        # ----------------------------------------------------

        Gst.init(None)

        # ----------------------------------------------------
        # Initialize OpenVINO
        # ----------------------------------------------------

        print("\nInitializing OpenVINO...")

        self.core = ov.Core()

        available_devices = self.core.available_devices

        print("Available OpenVINO devices:")

        for device in available_devices:
            print(f"  - {device}")

        if DEVICE not in available_devices:
            raise RuntimeError(
                f"{DEVICE} device is not available. "
                f"Available devices: {available_devices}"
            )

        # ----------------------------------------------------
        # Load OpenVINO model
        # ----------------------------------------------------

        print("\nLoading OpenVINO model...")

        self.model = self.core.read_model(
            MODEL_PATH
        )

        print("Model loaded successfully.")

        # ----------------------------------------------------
        # Print model information
        # ----------------------------------------------------

        print("\nModel input:")

        for input_layer in self.model.inputs:
            print(
                f"  Shape : {input_layer.shape}"
            )
            print(
                f"  Type  : {input_layer.element_type}"
            )

        print("\nModel output:")

        for output_layer in self.model.outputs:
            print(
                f"  Shape : {output_layer.shape}"
            )
            print(
                f"  Type  : {output_layer.element_type}"
            )

        # ----------------------------------------------------
        # Compile model for Intel GPU
        # ----------------------------------------------------

        print(
            f"\nCompiling model for Intel {DEVICE}..."
        )

        self.compiled_model = self.core.compile_model(
            self.model,
            DEVICE
        )

        self.infer_request = (
            self.compiled_model.create_infer_request()
        )

        print(
            "OpenVINO model compiled successfully."
        )

        print(
            f"Inference device: {DEVICE}"
        )

        # ----------------------------------------------------
        # Get model input/output ports
        # ----------------------------------------------------

        self.input_port = (
            self.compiled_model.input(0)
        )

        self.output_port = (
            self.compiled_model.output(0)
        )

        print(
            f"\nInput shape : {self.input_port.shape}"
        )

        print(
            f"Output shape: {self.output_port.shape}"
        )

        # ----------------------------------------------------
        # GStreamer pipeline
        # ----------------------------------------------------

        self.pipeline = Gst.Pipeline.new(
            "gstreamer-openvino-pipeline"
        )

        if self.pipeline is None:
            raise RuntimeError(
                "Failed to create GStreamer pipeline"
            )

        # ----------------------------------------------------
        # Create GStreamer elements
        # ----------------------------------------------------

        self.source = Gst.ElementFactory.make(
            "filesrc",
            "file-source"
        )

        self.demuxer = Gst.ElementFactory.make(
            "qtdemux",
            "demuxer"
        )

        self.queue = Gst.ElementFactory.make(
            "queue",
            "video-queue"
        )

        self.parser = Gst.ElementFactory.make(
            "h264parse",
            "h264-parser"
        )

        self.decoder = Gst.ElementFactory.make(
            "avdec_h264",
            "h264-decoder"
        )

        self.converter = Gst.ElementFactory.make(
            "videoconvert",
            "video-converter"
        )

        self.capsfilter = Gst.ElementFactory.make(
            "capsfilter",
            "video-caps"
        )

        self.sink = Gst.ElementFactory.make(
            "appsink",
            "video-sink"
        )

        elements = [
            self.source,
            self.demuxer,
            self.queue,
            self.parser,
            self.decoder,
            self.converter,
            self.capsfilter,
            self.sink,
        ]

        for element in elements:

            if element is None:
                raise RuntimeError(
                    "Failed to create one or more "
                    "GStreamer elements"
                )

        # ----------------------------------------------------
        # Configure source
        # ----------------------------------------------------

        self.source.set_property(
            "location",
            VIDEO_PATH
        )

        # ----------------------------------------------------
        # Configure BGR caps
        # ----------------------------------------------------

        caps = Gst.Caps.from_string(
            "video/x-raw,format=BGR"
        )

        self.capsfilter.set_property(
            "caps",
            caps
        )

        # ----------------------------------------------------
        # Configure appsink
        # ----------------------------------------------------

        self.sink.set_property(
            "emit-signals",
            True
        )

        self.sink.set_property(
            "sync",
            False
        )

        self.sink.set_property(
            "drop",
            True
        )

        self.sink.set_property(
            "max-buffers",
            1
        )

        self.sink.connect(
            "new-sample",
            self.on_new_sample
        )

        # ----------------------------------------------------
        # Add elements to pipeline
        # ----------------------------------------------------

        for element in elements:
            self.pipeline.add(element)

        # ----------------------------------------------------
        # Static linking
        # ----------------------------------------------------

        if not self.source.link(
            self.demuxer
        ):
            raise RuntimeError(
                "Failed to link filesrc → qtdemux"
            )

        if not self.queue.link(
            self.parser
        ):
            raise RuntimeError(
                "Failed to link queue → h264parse"
            )

        if not self.parser.link(
            self.decoder
        ):
            raise RuntimeError(
                "Failed to link h264parse → avdec_h264"
            )

        if not self.decoder.link(
            self.converter
        ):
            raise RuntimeError(
                "Failed to link avdec_h264 → videoconvert"
            )

        if not self.converter.link(
            self.capsfilter
        ):
            raise RuntimeError(
                "Failed to link videoconvert → capsfilter"
            )

        if not self.capsfilter.link(
            self.sink
        ):
            raise RuntimeError(
                "Failed to link capsfilter → appsink"
            )

        # ----------------------------------------------------
        # Dynamic qtdemux pad
        # ----------------------------------------------------

        self.demuxer.connect(
            "pad-added",
            self.on_pad_added
        )

        # ----------------------------------------------------
        # Runtime statistics
        # ----------------------------------------------------

        self.frame_count = 0
        self.inference_count = 0

        self.start_time = None

        print(
            "\nGStreamer pipeline created successfully."
        )


    # ========================================================
    # QTDemux PAD CALLBACK
    # ========================================================

    def on_pad_added(
        self,
        demuxer,
        pad
    ):

        print(
            f"\nNew pad detected: "
            f"{pad.get_name()}"
        )

        caps = pad.get_current_caps()

        if caps is None:
            caps = pad.query_caps(None)

        if caps is None:
            return

        structure = caps.get_structure(0)

        media_type = structure.get_name()

        print(
            f"  Media type: {media_type}"
        )

        # ----------------------------------------------------
        # We only want H264 video
        # ----------------------------------------------------

        if media_type != "video/x-h264":

            print(
                "  Ignoring non-H264 stream"
            )

            return

        sink_pad = (
            self.queue.get_static_pad(
                "sink"
            )
        )

        if sink_pad is None:

            print(
                "  ERROR: Could not get "
                "queue sink pad"
            )

            return

        if sink_pad.is_linked():

            print(
                "  Queue sink pad already linked"
            )

            return

        result = pad.link(
            sink_pad
        )

        if result == Gst.PadLinkReturn.OK:

            print(
                "  Successfully linked "
                "qtdemux → queue"
            )

        else:

            print(
                "  Failed to link "
                f"qtdemux → queue: {result}"
            )


    # ========================================================
    # PREPROCESSING
    # ========================================================

    def preprocess(
        self,
        frame
    ):
        """
        Convert GStreamer BGR frame into
        YOLO OpenVINO input tensor.

        Input:

            H x W x 3

        Output:

            1 x 3 x 640 x 640
        """

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        resized = cv2.resize(
            frame,
            (
                INPUT_WIDTH,
                INPUT_HEIGHT
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # uint8 → float32
        # ----------------------------------------------------

        image = rgb.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Normalize 0-255 → 0-1
        # ----------------------------------------------------

        image /= 255.0

        # ----------------------------------------------------
        # HWC → CHW
        # ----------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image = np.expand_dims(
            image,
            axis=0
        )

        return image


    # ========================================================
    # OPENVINO INFERENCE
    # ========================================================

    def run_inference(
        self,
        frame
    ):

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        input_tensor = self.preprocess(
            frame
        )

        # ----------------------------------------------------
        # Create OpenVINO Tensor
        # ----------------------------------------------------

        tensor = ov.Tensor(
            input_tensor
        )

        # ----------------------------------------------------
        # Set input tensor
        #
        # IMPORTANT:
        #
        # We are NOT using:
        #
        # infer_request.infer({
        #     self.input_port: input_tensor
        # })
        #
        # because the newer OpenVINO API can reject
        # Output objects as dictionary keys.
        #
        # Instead we explicitly set the tensor.
        # ----------------------------------------------------

        self.infer_request.set_input_tensor(
            tensor
        )

        # ----------------------------------------------------
        # Run inference
        # ----------------------------------------------------

        self.infer_request.infer()

        # ----------------------------------------------------
        # Get output tensor
        # ----------------------------------------------------

        output_tensor = (
            self.infer_request.get_output_tensor()
        )

        output = output_tensor.data

        return np.array(
            output
        )


    # ========================================================
    # YOLO POSTPROCESSING
    # ========================================================

    def postprocess(
        self,
        output,
        original_width,
        original_height
    ):

        # ----------------------------------------------------
        # Expected YOLO output:
        #
        # [1, 14, 8400]
        #
        # 14 =
        #
        # 4 bounding box values
        # +
        # 10 class scores
        #
        # ----------------------------------------------------

        predictions = output[0]

        # ----------------------------------------------------
        # Convert:
        #
        # [14, 8400]
        #
        # →
        #
        # [8400, 14]
        # ----------------------------------------------------

        predictions = np.transpose(
            predictions
        )

        boxes = []
        scores = []
        class_ids = []

        # ----------------------------------------------------
        # Scale factors
        # ----------------------------------------------------

        scale_x = (
            original_width /
            INPUT_WIDTH
        )

        scale_y = (
            original_height /
            INPUT_HEIGHT
        )

        # ----------------------------------------------------
        # Process predictions
        # ----------------------------------------------------

        for prediction in predictions:

            # -----------------------------------------------
            # YOLO format:
            #
            # cx
            # cy
            # width
            # height
            # class scores...
            # -----------------------------------------------

            cx = prediction[0]
            cy = prediction[1]
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

            # -----------------------------------------------
            # Convert center format → corner format
            # -----------------------------------------------

            x1 = (
                cx - width / 2
            ) * scale_x

            y1 = (
                cy - height / 2
            ) * scale_y

            x2 = (
                cx + width / 2
            ) * scale_x

            y2 = (
                cy + height / 2
            ) * scale_y

            # -----------------------------------------------
            # Clamp coordinates
            # -----------------------------------------------

            x1 = max(
                0,
                min(
                    original_width - 1,
                    int(x1)
                )
            )

            y1 = max(
                0,
                min(
                    original_height - 1,
                    int(y1)
                )
            )

            x2 = max(
                0,
                min(
                    original_width - 1,
                    int(x2)
                )
            )

            y2 = max(
                0,
                min(
                    original_height - 1,
                    int(y2)
                )
            )

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:
                continue

            boxes.append(
                [
                    x1,
                    y1,
                    box_width,
                    box_height
                ]
            )

            scores.append(
                confidence
            )

            class_ids.append(
                class_id
            )

        # ----------------------------------------------------
        # NMS
        # ----------------------------------------------------

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )

        detections = []

        if len(indices) == 0:
            return detections

        indices = np.array(
            indices
        ).reshape(-1)

        for index in indices:

            x, y, w, h = boxes[index]

            detections.append(
                {
                    "class_id": class_ids[index],
                    "class_name": CLASS_NAMES.get(
                        class_ids[index],
                        str(class_ids[index])
                    ),
                    "confidence": scores[index],
                    "bbox": (
                        x,
                        y,
                        w,
                        h
                    ),
                }
            )

        return detections


    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    def draw_detections(
        self,
        frame,
        detections
    ):

        for detection in detections:

            class_id = detection[
                "class_id"
            ]

            class_name = detection[
                "class_name"
            ]

            confidence = detection[
                "confidence"
            ]

            x, y, w, h = detection[
                "bbox"
            ]

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (
                    x,
                    y
                ),
                (
                    x + w,
                    y + h
                ),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x,
                    max(
                        20,
                        y - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        return frame


    # ========================================================
    # GSTREAMER APPSINK CALLBACK
    # ========================================================

    def on_new_sample(
        self,
        sink
    ):

        sample = sink.emit(
            "pull-sample"
        )

        if sample is None:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()

        caps = sample.get_caps()

        if buffer is None or caps is None:
            return Gst.FlowReturn.ERROR

        structure = caps.get_structure(0)

        width = structure.get_value(
            "width"
        )

        height = structure.get_value(
            "height"
        )

        # ----------------------------------------------------
        # Map GStreamer buffer
        # ----------------------------------------------------

        success, map_info = buffer.map(
            Gst.MapFlags.READ
        )

        if not success:

            print(
                "Failed to map GStreamer buffer"
            )

            return Gst.FlowReturn.ERROR

        try:

            # ------------------------------------------------
            # Convert buffer → NumPy
            # ------------------------------------------------

            frame = np.frombuffer(
                map_info.data,
                dtype=np.uint8
            )

            frame = frame.reshape(
                (
                    height,
                    width,
                    3
                )
            )

            # ------------------------------------------------
            # Frame counter
            # ------------------------------------------------

            self.frame_count += 1

            if self.start_time is None:
                self.start_time = time.time()

            # ------------------------------------------------
            # OpenVINO inference
            # ------------------------------------------------

            inference_start = time.perf_counter()

            output = self.run_inference(
                frame
            )

            inference_end = time.perf_counter()

            inference_time = (
                inference_end -
                inference_start
            )

            self.inference_count += 1

            # ------------------------------------------------
            # YOLO postprocessing
            # ------------------------------------------------

            detections = self.postprocess(
                output,
                width,
                height
            )

            # ------------------------------------------------
            # Draw detections
            # ------------------------------------------------

            display_frame = frame.copy()

            display_frame = (
                self.draw_detections(
                    display_frame,
                    detections
                )
            )

            # ------------------------------------------------
            # Display FPS
            # ------------------------------------------------

            elapsed = (
                time.time() -
                self.start_time
            )

            fps = (
                self.frame_count /
                elapsed
                if elapsed > 0
                else 0
            )

            cv2.putText(
                display_frame,
                f"FPS: {fps:.2f}",
                (
                    20,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                (
                    f"Inference: "
                    f"{inference_time * 1000:.2f} ms"
                ),
                (
                    20,
                    60
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                (
                    f"Detections: "
                    f"{len(detections)}"
                ),
                (
                    20,
                    90
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Print detection information
            # ------------------------------------------------

            if detections:

                print(
                    f"\nFrame "
                    f"{self.frame_count}: "
                    f"{len(detections)} detections"
                )

                for detection in detections:

                    print(
                        f"  "
                        f"{detection['class_name']}: "
                        f"{detection['confidence']:.2f}"
                    )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(
                "GStreamer + OpenVINO PPE",
                display_frame
            )

            key = (
                cv2.waitKey(1) &
                0xFF
            )

            if key == ord("q"):

                print(
                    "\nQ pressed - "
                    "stopping pipeline"
                )

                self.pipeline.send_event(
                    Gst.Event.new_eos()
                )

        except Exception as exc:

            print(
                f"\nInference error: {exc}"
            )

        finally:

            buffer.unmap(
                map_info
            )

        return Gst.FlowReturn.OK


    # ========================================================
    # GSTREAMER BUS
    # ========================================================

    def on_message(
        self,
        bus,
        message,
        loop
    ):

        message_type = message.type

        if message_type == Gst.MessageType.ERROR:

            error, debug = (
                message.parse_error()
            )

            print(
                "\nGSTREAMER ERROR:"
            )

            print(error)

            if debug:
                print(
                    f"DEBUG: {debug}"
                )

            loop.quit()

        elif message_type == Gst.MessageType.EOS:

            print(
                "\nEnd of stream (EOS)"
            )

            loop.quit()

        elif message_type == Gst.MessageType.STATE_CHANGED:

            if message.src == self.pipeline:

                old_state, new_state, pending = (
                    message.parse_state_changed()
                )

                print(
                    "Pipeline state changed: "
                    f"{old_state.value_nick} "
                    "→ "
                    f"{new_state.value_nick}"
                )


    # ========================================================
    # RUN PIPELINE
    # ========================================================

    def run(self):

        print(
            "\nStarting GStreamer pipeline..."
        )

        loop = GLib.MainLoop()

        bus = self.pipeline.get_bus()

        bus.add_signal_watch()

        bus.connect(
            "message",
            self.on_message,
            loop
        )

        result = self.pipeline.set_state(
            Gst.State.PLAYING
        )

        if result == Gst.StateChangeReturn.FAILURE:

            print(
                "Failed to set pipeline "
                "to PLAYING"
            )

            self.pipeline.set_state(
                Gst.State.NULL
            )

            return

        print(
            "Pipeline is PLAYING"
        )

        try:

            loop.run()

        except KeyboardInterrupt:

            print(
                "\nKeyboard interrupt received"
            )

        finally:

            print(
                "\nStopping pipeline..."
            )

            self.pipeline.set_state(
                Gst.State.NULL
            )

            cv2.destroyAllWindows()

            print(
                "Pipeline stopped"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        application = (
            GStreamerOpenVINO()
        )

        application.run()

    except Exception as exc:

        print(
            f"\nFatal error: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()