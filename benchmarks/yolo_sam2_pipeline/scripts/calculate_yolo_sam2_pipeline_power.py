import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np


ACTIVE_LOG = Path(
    "benchmark_results/"
    "yolo_sam2_offline_bf16_active_15w.log"
)

RUN_JSON = Path(
    "benchmark_results/"
    "yolo_fp32_sam2_bf16_offline_pipeline_benchmark.json"
)

OUTPUT_JSON = Path(
    "benchmark_results/"
    "yolo_fp32_sam2_bf16_offline_pipeline_power_summary.json"
)

# Same-day 15 W idle baseline measured previously with 531 samples.
IDLE_SAMPLES = 531
IDLE_MEAN_W = 5.248521657250471
IDLE_STD_W = 0.04507491669414587
IDLE_MIN_W = 5.156
IDLE_MEDIAN_W = 5.234
IDLE_MAX_W = 5.431


timestamp_pattern = re.compile(
    r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})"
)

power_pattern = re.compile(
    r"\bVDD_IN\s+(\d+)mW/"
)


def parse_iso_datetime(value):
    return datetime.fromisoformat(value)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)

    return {
        "samples": int(len(array)),
        "mean_w": float(np.mean(array)),
        "sample_std_w": (
            float(np.std(array, ddof=1))
            if len(array) > 1
            else 0.0
        ),
        "min_w": float(np.min(array)),
        "median_w": float(np.median(array)),
        "max_w": float(np.max(array)),
    }


with open(RUN_JSON) as file:
    run_data = json.load(file)

run = run_data["run"]

start_exact = parse_iso_datetime(
    run["start_time"]
)
end_exact = parse_iso_datetime(
    run["end_time"]
)

# Discard partial first and final seconds.
start_inclusive = start_exact.replace(
    microsecond=0
)

if start_exact.microsecond > 0:
    start_inclusive = start_inclusive.replace(
        second=start_inclusive.second + 1
    )

end_exclusive = end_exact.replace(
    microsecond=0
)

active_power_values = []

with open(ACTIVE_LOG) as file:
    for line in file:
        timestamp_match = timestamp_pattern.search(line)
        power_match = power_pattern.search(line)

        if timestamp_match is None:
            continue

        if power_match is None:
            continue

        sample_time = datetime.strptime(
            timestamp_match.group(1),
            "%m-%d-%Y %H:%M:%S",
        )

        if (
            start_inclusive
            <= sample_time
            < end_exclusive
        ):
            power_w = (
                float(power_match.group(1))
                / 1000.0
            )

            active_power_values.append(power_w)

if not active_power_values:
    raise RuntimeError(
        "No VDD_IN samples were found in the active window"
    )

active_power = summarize(active_power_values)

idle_power = {
    "samples": IDLE_SAMPLES,
    "mean_w": IDLE_MEAN_W,
    "sample_std_w": IDLE_STD_W,
    "min_w": IDLE_MIN_W,
    "median_w": IDLE_MEDIAN_W,
    "max_w": IDLE_MAX_W,
}

duration_s = float(run["duration_s"])
inference_count = int(run["inference_count"])
throughput_fps = float(run["throughput_fps"])

active_mean_w = active_power["mean_w"]
incremental_power_w = (
    active_mean_w - IDLE_MEAN_W
)

gross_total_energy_j = (
    active_mean_w * duration_s
)

idle_equivalent_energy_j = (
    IDLE_MEAN_W * duration_s
)

incremental_total_energy_j = (
    incremental_power_w * duration_s
)

gross_energy_per_inference_j = (
    gross_total_energy_j
    / inference_count
)

incremental_energy_per_inference_j = (
    incremental_total_energy_j
    / inference_count
)

gross_inferences_per_joule = (
    inference_count
    / gross_total_energy_j
)

incremental_inferences_per_joule = (
    inference_count
    / incremental_total_energy_j
)

fps_per_active_watt = (
    throughput_fps
    / active_mean_w
)

result = {
    "pipeline": (
        "YOLOv8n FP32 + "
        "SAM2 Hiera Tiny BF16"
    ),
    "power_mode": "15W",
    "power_rail": "VDD_IN",
    "active_log": str(ACTIVE_LOG),
    "active_window": {
        "start_inclusive": (
            start_inclusive.isoformat()
        ),
        "end_exclusive": (
            end_exclusive.isoformat()
        ),
    },
    "idle_power": idle_power,
    "active_power": active_power,
    "incremental_power_w": (
        incremental_power_w
    ),
    "run": run,
    "energy": {
        "gross_total_energy_j": (
            gross_total_energy_j
        ),
        "idle_equivalent_energy_j": (
            idle_equivalent_energy_j
        ),
        "incremental_total_energy_j": (
            incremental_total_energy_j
        ),
        "gross_energy_per_inference_j": (
            gross_energy_per_inference_j
        ),
        "incremental_energy_per_inference_j": (
            incremental_energy_per_inference_j
        ),
        "gross_inferences_per_joule": (
            gross_inferences_per_joule
        ),
        "incremental_inferences_per_joule": (
            incremental_inferences_per_joule
        ),
        "fps_per_active_watt": (
            fps_per_active_watt
        ),
    },
    "method": {
        "tegrastats_interval_ms": 100,
        "instantaneous_vdd_in_used": True,
        "common_same_day_idle_baseline_used": True,
        "idle_baseline_samples": IDLE_SAMPLES,
        "boundary_seconds_discarded": True,
        "camera_capture_included": False,
        "display_included": False,
        "yolo_included": True,
        "sam2_image_encoding_included": True,
        "sam2_prompt_and_decoder_included": True,
    },
}

with open(OUTPUT_JSON, "w") as file:
    json.dump(result, file, indent=2)

print(json.dumps(result, indent=2))
print()
print(f"Saved to: {OUTPUT_JSON}")