import sys
import gi
import numpy as np
import cv2

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VIDEO_PATH = r"D:\ppe_monitoring\data\input_videos\factory.mp4"


class GStreamerPipeline:

    def __init__(self):

        # -------------------------------------------------
        # Initialize GStreamer
        # -------------------------------------------------

        Gst.init(None)

        self.pipeline = Gst.Pipeline.new(
            "ppe-video-pipeline"
        )

        if self.pipeline is None:
            raise RuntimeError(
                "Failed to create GStreamer pipeline"
            )

        # -------------------------------------------------
        # Create GStreamer elements
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Capsfilter
        #
        # Force GStreamer to provide BGR frames.
        # OpenCV uses BGR by default.
        # -------------------------------------------------

        self.video_caps = Gst.ElementFactory.make(
            "capsfilter",
            "video-caps"
        )

        self.sink = Gst.ElementFactory.make(
            "appsink",
            "video-sink"
        )

        # -------------------------------------------------
        # Validate elements
        # -------------------------------------------------

        elements = [
            self.source,
            self.demuxer,
            self.queue,
            self.parser,
            self.decoder,
            self.converter,
            self.video_caps,
            self.sink,
        ]

        for element in elements:

            if element is None:
                raise RuntimeError(
                    "Failed to create one or more "
                    "GStreamer elements"
                )

        # -------------------------------------------------
        # Configure filesrc
        # -------------------------------------------------

        self.source.set_property(
            "location",
            VIDEO_PATH
        )

        # -------------------------------------------------
        # Configure capsfilter
        # -------------------------------------------------

        caps = Gst.Caps.from_string(
            "video/x-raw,format=BGR"
        )

        self.video_caps.set_property(
            "caps",
            caps
        )

        # -------------------------------------------------
        # Configure appsink
        # -------------------------------------------------

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

        # Connect appsink signal
        self.sink.connect(
            "new-sample",
            self.on_new_sample
        )

        # -------------------------------------------------
        # Add elements to pipeline
        # -------------------------------------------------

        for element in elements:
            self.pipeline.add(element)

        # -------------------------------------------------
        # Link static elements
        # -------------------------------------------------

        # filesrc → qtdemux
        if not self.source.link(self.demuxer):

            raise RuntimeError(
                "Failed to link filesrc → qtdemux"
            )

        # queue → h264parse
        if not self.queue.link(self.parser):

            raise RuntimeError(
                "Failed to link queue → h264parse"
            )

        # h264parse → avdec_h264
        if not self.parser.link(self.decoder):

            raise RuntimeError(
                "Failed to link h264parse → avdec_h264"
            )

        # avdec_h264 → videoconvert
        if not self.decoder.link(self.converter):

            raise RuntimeError(
                "Failed to link avdec_h264 → videoconvert"
            )

        # videoconvert → capsfilter
        if not self.converter.link(self.video_caps):

            raise RuntimeError(
                "Failed to link "
                "videoconvert → capsfilter"
            )

        # capsfilter → appsink
        if not self.video_caps.link(self.sink):

            raise RuntimeError(
                "Failed to link "
                "capsfilter → appsink"
            )

        # -------------------------------------------------
        # qtdemux creates pads dynamically
        # -------------------------------------------------

        self.demuxer.connect(
            "pad-added",
            self.on_pad_added
        )

        print(
            "GStreamer pipeline created successfully"
        )

    # =====================================================
    # qtdemux dynamic pad callback
    # =====================================================

    def on_pad_added(
        self,
        demuxer,
        pad
    ):

        print(
            f"New pad detected: {pad.get_name()}"
        )

        # -------------------------------------------------
        # Get caps
        # -------------------------------------------------

        caps = pad.get_current_caps()

        if caps is None:
            caps = pad.query_caps(None)

        if caps is None or caps.get_size() == 0:

            print(
                "  ERROR: Could not determine pad caps"
            )

            return

        # -------------------------------------------------
        # Get media type
        # -------------------------------------------------

        structure = caps.get_structure(0)

        media_type = structure.get_name()

        print(
            f"  Media type: {media_type}"
        )

        # -------------------------------------------------
        # Only accept H264 video
        # -------------------------------------------------

        if media_type != "video/x-h264":

            print(
                "  Ignoring non-H264 stream"
            )

            return

        # -------------------------------------------------
        # Get queue sink pad
        # -------------------------------------------------

        sink_pad = self.queue.get_static_pad(
            "sink"
        )

        if sink_pad is None:

            print(
                "  ERROR: Could not get queue sink pad"
            )

            return

        # -------------------------------------------------
        # Prevent duplicate linking
        # -------------------------------------------------

        if sink_pad.is_linked():

            print(
                "  Queue sink pad is already linked"
            )

            return

        # -------------------------------------------------
        # Link qtdemux → queue
        # -------------------------------------------------

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

    # =====================================================
    # GStreamer bus message handler
    # =====================================================

    def on_message(
        self,
        bus,
        message,
        loop
    ):

        message_type = message.type

        # -------------------------------------------------
        # ERROR
        # -------------------------------------------------

        if message_type == Gst.MessageType.ERROR:

            error, debug = message.parse_error()

            print(
                "\nERROR:"
            )

            print(error)

            if debug:

                print(
                    f"DEBUG: {debug}"
                )

            loop.quit()

        # -------------------------------------------------
        # EOS
        # -------------------------------------------------

        elif message_type == Gst.MessageType.EOS:

            print(
                "\nEnd of stream (EOS)"
            )

            loop.quit()

        # -------------------------------------------------
        # Pipeline state changes
        # -------------------------------------------------

        elif (
            message_type
            == Gst.MessageType.STATE_CHANGED
        ):

            if message.src == self.pipeline:

                (
                    old_state,
                    new_state,
                    pending_state
                ) = message.parse_state_changed()

                print(
                    "Pipeline state changed: "
                    f"{old_state.value_nick} → "
                    f"{new_state.value_nick}"
                )

    # =====================================================
    # appsink callback
    # =====================================================

    def on_new_sample(
        self,
        sink
    ):

        # -------------------------------------------------
        # Pull sample from appsink
        # -------------------------------------------------

        sample = sink.emit(
            "pull-sample"
        )

        if sample is None:

            return Gst.FlowReturn.ERROR

        # -------------------------------------------------
        # Get Gst.Buffer
        # -------------------------------------------------

        buffer = sample.get_buffer()

        # -------------------------------------------------
        # Get caps
        # -------------------------------------------------

        caps = sample.get_caps()

        if buffer is None or caps is None:

            return Gst.FlowReturn.ERROR

        # -------------------------------------------------
        # Get video dimensions
        # -------------------------------------------------

        structure = caps.get_structure(0)

        width = structure.get_value(
            "width"
        )

        height = structure.get_value(
            "height"
        )

        # -------------------------------------------------
        # Map Gst.Buffer
        # -------------------------------------------------

        success, map_info = buffer.map(
            Gst.MapFlags.READ
        )

        if not success:

            print(
                "Failed to map GStreamer buffer"
            )

            return Gst.FlowReturn.ERROR

        try:

            # -------------------------------------------------
            # Convert raw buffer to NumPy array
            # -------------------------------------------------

            frame = np.frombuffer(
                map_info.data,
                dtype=np.uint8
            )

            # -------------------------------------------------
            # Reshape into BGR image
            # -------------------------------------------------

            frame = frame.reshape(
                (height, width, 3)
            )

            # -------------------------------------------------
            # Display using OpenCV
            # -------------------------------------------------

            cv2.imshow(
                "GStreamer PPE Video",
                frame
            )

            # -------------------------------------------------
            # Process keyboard input
            # -------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                print(
                    "\nQ pressed - stopping pipeline"
                )

                self.pipeline.send_event(
                    Gst.Event.new_eos()
                )

        except Exception as exc:

            print(
                f"Frame processing error: {exc}"
            )

            return Gst.FlowReturn.ERROR

        finally:

            # -------------------------------------------------
            # Always unmap buffer
            # -------------------------------------------------

            buffer.unmap(
                map_info
            )

        return Gst.FlowReturn.OK

    # =====================================================
    # Run pipeline
    # =====================================================

    def run(self):

        print(
            "\nStarting pipeline..."
        )

        # -------------------------------------------------
        # Create GLib main loop
        # -------------------------------------------------

        loop = GLib.MainLoop()

        # -------------------------------------------------
        # Get GStreamer bus
        # -------------------------------------------------

        bus = self.pipeline.get_bus()

        bus.add_signal_watch()

        bus.connect(
            "message",
            self.on_message,
            loop
        )

        # -------------------------------------------------
        # Start pipeline
        # -------------------------------------------------

        result = self.pipeline.set_state(
            Gst.State.PLAYING
        )

        if (
            result
            == Gst.StateChangeReturn.FAILURE
        ):

            print(
                "Failed to set pipeline to PLAYING"
            )

            self.pipeline.set_state(
                Gst.State.NULL
            )

            return

        print(
            "Pipeline is PLAYING"
        )

        # -------------------------------------------------
        # Run main loop
        # -------------------------------------------------

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

            # Close OpenCV windows
            cv2.destroyAllWindows()

            print(
                "Pipeline stopped"
            )


# =========================================================
# Main
# =========================================================

def main():

    try:

        pipeline = GStreamerPipeline()

        pipeline.run()

    except Exception as exc:

        print(
            f"\nFatal error: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":

    main()