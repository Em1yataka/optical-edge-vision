import csv
from pathlib import Path
from datetime import datetime


class CSVLogger:
    def __init__(self, log_path="logs/image_quality_log.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(exist_ok=True)

        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time",
                    "fps",
                    "brightness",
                    "brightness_status",
                    "contrast",
                    "contrast_status",
                    "blur",
                    "blur_status",
                    "objects",
                    "sam2_score",
                    "mask_area",
                    "sam2_time_ms",
                ])

    def log(
        self,
        fps,
        quality,
        counts,
        sam2_score=None,
        mask_area=None,
        sam2_time_ms=None,
    ):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(fps, 2),
                round(quality["brightness"], 2),
                quality["brightness_status"],
                round(quality["contrast"], 2),
                quality["contrast_status"],
                round(quality["blur"], 2),
                quality["blur_status"],
                dict(counts),
                round(sam2_score, 4) if sam2_score is not None else "",
                mask_area if mask_area is not None else "",
                round(sam2_time_ms, 2) if sam2_time_ms is not None else "",
            ])