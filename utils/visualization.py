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


def draw_quality(frame, quality):
    y = 180

    cv2.putText(
        frame,
        f"Brightness: {quality['brightness']:.1f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    y += 30

    cv2.putText(
        frame,
        f"Contrast: {quality['contrast']:.1f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    y += 30

    cv2.putText(
        frame,
        f"Blur: {quality['blur']:.1f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )