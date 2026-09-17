import cv2
from collections import defaultdict, deque

from modules.camera import Camera
from modules.detector import Detector
from modules.qwen_vision import QwenVision
from modules.qwen_worker import QwenWorker
from modules.speaker import Speaker
from modules.ai_engine import AIEngine

from bytetrack_wrapper import MultiClassByteTracker


# ============================================================
# PROJECT START
# ============================================================

print("===================================")
print("       BlindAssist AI Vision")
print("===================================")


# ============================================================
# CAMERA
# ============================================================

print("Starting camera...")

camera = Camera()


# ============================================================
# YOLO26s
# ============================================================

print("Starting YOLO26s...")

detector = Detector()

print("YOLO26s ready.")


# ============================================================
# BYTETrack
# ============================================================

print("Starting ByteTrack...")

tracker = MultiClassByteTracker(
    class_names=list(detector.model.names.values()),
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    low_thresh=0.1,
)

print("ByteTrack ready.")


# ============================================================
# TRACKING HISTORY
# ============================================================

track_history = defaultdict(
    lambda: deque(maxlen=20)
)


# ============================================================
# AI ENGINE
# ============================================================

print("Starting AI Engine...")

ai_engine = AIEngine()

print("AI Engine ready.")


# ============================================================
# QWEN VISION
# ============================================================

print("Starting Qwen...")

qwen = QwenVision()


# ============================================================
# SPEAKER
# ============================================================

print("Starting speaker...")

speaker = Speaker()


# ============================================================
# QWEN BACKGROUND WORKER
# ============================================================

worker = QwenWorker(
    qwen,
    interval=4.0
)


# ============================================================
# READY
# ============================================================

print()
print("BlindAssist is ready.")
print("Press Q to quit.")
print()


# ============================================================
# YOLO → BYTETrack CONVERSION
# ============================================================

def yolo_to_tracker_detections(results, detector):

    detections = []

    result = results[0]

    if result.boxes is None:
        return detections

    boxes = result.boxes

    for i in range(len(boxes)):

        # Bounding box
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()

        # Confidence
        confidence = float(
            boxes.conf[i]
        )

        # Class ID
        class_id = int(
            boxes.cls[i]
        )

        # Class name
        class_name = detector.model.names[
            class_id
        ]

        detections.append(
            {
                "class_name": class_name,

                "class_id": class_id,

                "confidence": confidence,

                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            }
        )

    return detections


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # GET CAMERA FRAME
        # ----------------------------------------------------

        frame = camera.get_frame()

        if frame is None:

            print("Camera frame unavailable.")

            break


        # ----------------------------------------------------
        # SEND NEWEST FRAME TO QWEN
        # ----------------------------------------------------

        worker.update_frame(frame)


        # ----------------------------------------------------
        # YOLO26s DETECTION
        # ----------------------------------------------------

        results = detector.detect(frame)


        # ----------------------------------------------------
        # CONVERT YOLO DETECTIONS
        # ----------------------------------------------------

        detections = yolo_to_tracker_detections(
            results,
            detector
        )


        # ----------------------------------------------------
        # BYTETrack
        # ----------------------------------------------------

        tracks = tracker.update(
            detections
        )


        # ----------------------------------------------------
        # DRAW YOLO DETECTIONS
        # ----------------------------------------------------

        display_frame = results[0].plot()


        # ----------------------------------------------------
        # DRAW BYTETrack OUTPUT
        # ----------------------------------------------------

        for track in tracks:

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                track["bbox"]
            )

            # Object information
            class_name = track["class_name"]

            track_id = track["track_id"]


            # ------------------------------------------------
            # CENTER POINT
            # ------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # ------------------------------------------------
            # SAVE TRACK HISTORY
            # ------------------------------------------------

            track_history[
                track_id
            ].append(
                (
                    center_x,
                    center_y
                )
            )


            # ------------------------------------------------
            # DRAW TRACKING BOX
            # ------------------------------------------------

            cv2.rectangle(
                display_frame,

                (x1, y1),

                (x2, y2),

                (255, 0, 0),

                2
            )


            # ------------------------------------------------
            # DRAW TRACK ID
            # ------------------------------------------------

            label = (
                f"{class_name} "
                f"ID:{track_id}"
            )

            cv2.putText(
                display_frame,

                label,

                (
                    x1,
                    max(
                        20,
                        y1 - 10
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 0, 0),

                2
            )


            # ------------------------------------------------
            # DRAW CENTER POINT
            # ------------------------------------------------

            cv2.circle(
                display_frame,

                (
                    center_x,
                    center_y
                ),

                5,

                (0, 0, 255),

                -1
            )


            # ------------------------------------------------
            # DRAW MOVEMENT TRAIL
            # ------------------------------------------------

            points = track_history[
                track_id
            ]

            for i in range(
                1,
                len(points)
            ):

                cv2.line(
                    display_frame,

                    points[i - 1],

                    points[i],

                    (255, 0, 0),

                    2
                )


        # ----------------------------------------------------
        # TRACKING STATISTICS
        # ----------------------------------------------------

        cv2.putText(
            display_frame,

            f"Objects Detected: {len(detections)}",

            (20, 135),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 0),

            2
        )


        cv2.putText(
            display_frame,

            f"Active Tracks: {len(tracks)}",

            (20, 165),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 0),

            2
        )


        # ----------------------------------------------------
        # YOLO AI MESSAGE
        # ----------------------------------------------------

        try:

            yolo_message = (
                ai_engine.generate_message(
                    results,
                    detector
                )
            )

        except Exception as e:

            yolo_message = ""

            print(
                "[YOLO AI ERROR]:",
                e
            )


        # ----------------------------------------------------
        # PROJECT TITLE
        # ----------------------------------------------------

        cv2.putText(
            display_frame,

            "BlindAssist AI Vision",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )


        # ----------------------------------------------------
        # YOLO STATUS
        # ----------------------------------------------------

        cv2.putText(
            display_frame,

            "YOLO26s: Active",

            (20, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 0),

            2
        )


        # ----------------------------------------------------
        # QWEN STATUS
        # ----------------------------------------------------

        cv2.putText(
            display_frame,

            "Qwen-VL: Active",

            (20, 100),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 0),

            2
        )


        # ----------------------------------------------------
        # BYTETrack STATUS
        # ----------------------------------------------------

        cv2.putText(
            display_frame,

            "ByteTrack: Active",

            (20, 200),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 0, 0),

            2
        )


        # ----------------------------------------------------
        # GET QWEN DESCRIPTION
        # ----------------------------------------------------

        description = worker.get_description()


        # ----------------------------------------------------
        # SHOW CAMERA WINDOW
        # ----------------------------------------------------

        cv2.imshow(
            "BlindAssist AI Vision",

            display_frame
        )


        # ----------------------------------------------------
        # QWEN SPEECH
        # ----------------------------------------------------

        if description:

            print(
                "\n=============================="
            )

            print(
                "QWEN AI MESSAGE:"
            )

            print(
                description
            )

            print(
                "=============================="
            )

            speaker.speak(
                description
            )


        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    print(
        "\nStopping BlindAssist..."
    )

    worker.stop()

    camera.release()

    cv2.destroyAllWindows()

    print(
        "BlindAssist stopped."
    )