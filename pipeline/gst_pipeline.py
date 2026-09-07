import sys
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


VIDEO_PATH = r"D:\ppe_monitoring\data\input_videos\factory.mp4"


class GStreamerPipeline:
    def __init__(self):
        # Initialize GStreamer
        Gst.init(None)

        self.pipeline = Gst.Pipeline.new("ppe-video-pipeline")

        if self.pipeline is None:
            raise RuntimeError("Failed to create GStreamer pipeline")

        # -------------------------------------------------
        # Create GStreamer elements
        # -------------------------------------------------

        self.source = Gst.ElementFactory.make("filesrc", "file-source")
        self.demuxer = Gst.ElementFactory.make("qtdemux", "demuxer")
        self.queue = Gst.ElementFactory.make("queue", "video-queue")
        self.parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        self.decoder = Gst.ElementFactory.make("avdec_h264", "h264-decoder")
        self.converter = Gst.ElementFactory.make(
            "videoconvert",
            "video-converter"
        )
        self.sink = Gst.ElementFactory.make("autovideosink", "video-sink")

        elements = [
            self.source,
            self.demuxer,
            self.queue,
            self.parser,
            self.decoder,
            self.converter,
            self.sink,
        ]

        for element in elements:
            if element is None:
                raise RuntimeError(
                    "Failed to create one or more GStreamer elements"
                )

        # -------------------------------------------------
        # Configure elements
        # -------------------------------------------------

        self.source.set_property("location", VIDEO_PATH)

        # -------------------------------------------------
        # Add elements to pipeline
        # -------------------------------------------------

        for element in elements:
            self.pipeline.add(element)

        # -------------------------------------------------
        # Link static elements
        # -------------------------------------------------

        if not self.source.link(self.demuxer):
            raise RuntimeError(
                "Failed to link filesrc → qtdemux"
            )

        if not self.queue.link(self.parser):
            raise RuntimeError(
                "Failed to link queue → h264parse"
            )

        if not self.parser.link(self.decoder):
            raise RuntimeError(
                "Failed to link h264parse → avdec_h264"
            )

        if not self.decoder.link(self.converter):
            raise RuntimeError(
                "Failed to link avdec_h264 → videoconvert"
            )

        if not self.converter.link(self.sink):
            raise RuntimeError(
                "Failed to link videoconvert → autovideosink"
            )

        # -------------------------------------------------
        # qtdemux creates pads dynamically
        # -------------------------------------------------

        self.demuxer.connect(
            "pad-added",
            self.on_pad_added
        )

        print("GStreamer pipeline created successfully")

    # -----------------------------------------------------
    # Handle dynamically created qtdemux pads
    # -----------------------------------------------------

    def on_pad_added(self, demuxer, pad):
        print(f"New pad detected: {pad.get_name()}")

        caps = pad.get_current_caps()

        if caps is None:
            caps = pad.query_caps(None)

        structure = caps.get_structure(0)
        media_type = structure.get_name()

        print(f"  Media type: {media_type}")

        # We only want the H264 video stream.
        if media_type != "video/x-h264":
            print("  Ignoring non-H264 stream")
            return

        sink_pad = self.queue.get_static_pad("sink")

        if sink_pad is None:
            print("  ERROR: Could not get queue sink pad")
            return

        if sink_pad.is_linked():
            print("  Queue sink pad is already linked")
            return

        result = pad.link(sink_pad)

        if result == Gst.PadLinkReturn.OK:
            print("  Successfully linked qtdemux → queue")
        else:
            print(
                f"  Failed to link qtdemux → queue: {result}"
            )

    # -----------------------------------------------------
    # Handle GStreamer bus messages
    # -----------------------------------------------------

    def on_message(self, bus, message, loop):
        message_type = message.type

        if message_type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()

            print("\nERROR:")
            print(error)

            if debug:
                print(f"DEBUG: {debug}")

            loop.quit()

        elif message_type == Gst.MessageType.EOS:
            print("\nEnd of stream (EOS)")
            loop.quit()

        elif message_type == Gst.MessageType.STATE_CHANGED:

            if message.src == self.pipeline:

                old_state, new_state, pending_state = (
                    message.parse_state_changed()
                )

                print(
                    f"Pipeline state changed: "
                    f"{old_state.value_nick} → "
                    f"{new_state.value_nick}"
                )

    # -----------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------

    def run(self):

        print("\nStarting pipeline...")

        # Create GLib main loop
        loop = GLib.MainLoop()

        # Get pipeline bus
        bus = self.pipeline.get_bus()

        bus.add_signal_watch()

        bus.connect(
            "message",
            self.on_message,
            loop
        )

        # Start pipeline
        result = self.pipeline.set_state(
            Gst.State.PLAYING
        )

        if result == Gst.StateChangeReturn.FAILURE:
            print("Failed to set pipeline to PLAYING")
            self.pipeline.set_state(Gst.State.NULL)
            return

        print("Pipeline is PLAYING")

        try:
            loop.run()

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")

        finally:
            print("\nStopping pipeline...")

            self.pipeline.set_state(
                Gst.State.NULL
            )

            print("Pipeline stopped")


def main():
    try:
        pipeline = GStreamerPipeline()
        pipeline.run()

    except Exception as exc:
        print(f"\nFatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()