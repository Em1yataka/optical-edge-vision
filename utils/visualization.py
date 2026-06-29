import cv2


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )


def draw_counts(frame, counts):
    y = 80

    for name, count in counts.items():
        cv2.putText(
            frame,
            f"{name}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        y += 35