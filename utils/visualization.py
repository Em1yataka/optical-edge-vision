import cv2


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def draw_counts(frame, counts):
    y = 70
    font_scale = 0.5
    thickness = 1
    color = (0, 255, 255)

    for name, count in counts.items():
        cv2.putText(
            frame,
            f"{name}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )
        y += 25


def draw_quality(frame, quality):
    h, w = frame.shape[:2]

    x = 20
    y = h - 105

    font_scale = 0.45
    thickness = 1
    color = (255, 255, 0)

    lines = [
        f"Brightness : {quality['brightness']:.1f} ({quality['brightness_status']})",
        f"Contrast   : {quality['contrast']:.1f} ({quality['contrast_status']})",
        f"Blur       : {quality['blur']:.1f}",
        f"Sharpness  : {quality['blur_status']}",
    ]

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )
        y += 25