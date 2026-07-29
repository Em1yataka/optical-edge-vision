import datetime
import json
import math
import re
import statistics
from pathlib import Path


BENCHMARK_JSON = Path(
    "benchmark_results/sam3_bf16_336_benchmark.json"
)
IDLE_LOG = Path(
    "benchmark_results/sam3_idle_clean_15w_20260727.log"
)
ACTIVE_LOG = Path(
    "benchmark_results/sam3_active_clean_15w_20260727.log"
)
OUTPUT_JSON = Path(
    "benchmark_results/sam3_bf16_336_power_summary.json"
)

TIMESTAMP_PATTERN = re.compile(
    r"^(?P<timestamp>"
    r"\d{2}-\d{2}-\d{4} "
    r"\d{2}:\d{2}:\d{2}"
    r")"
)
VDD_IN_PATTERN = re.compile(
    r"\bVDD_IN "
    r"(?P<instant>\d+)mW/"
    r"(?P<running_average>\d+)mW"
)
RAM_PATTERN = re.compile(
    r"\bRAM "
    r"(?P<used>\d+)/(?P<total>\d+)MB"
)
GPU_TEMPERATURE_PATTERN = re.compile(
    r"\bgpu@(?P<temperature>\d+(?:\.\d+)?)C"
)
GR3D_PATTERN = re.compile(
    r"\bGR3D_FREQ "
    r"(?P<utilization>\d+)%"
)


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)


def parse_log(path):
    samples = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        for line_number, line in enumerate(file, 1):
            timestamp_match = TIMESTAMP_PATTERN.search(line)
            power_match = VDD_IN_PATTERN.search(line)

            if timestamp_match is None or power_match is None:
                continue

            timestamp = datetime.datetime.strptime(
                timestamp_match.group("timestamp"),
                "%m-%d-%Y %H:%M:%S",
            )

            ram_match = RAM_PATTERN.search(line)
            temperature_match = (
                GPU_TEMPERATURE_PATTERN.search(line)
            )
            gr3d_match = GR3D_PATTERN.search(line)

            sample = {
                "timestamp": timestamp,
                "line_number": line_number,
                "vdd_in_w": (
                    int(power_match.group("instant"))
                    / 1000.0
                ),
                "reported_running_average_w": (
                    int(
                        power_match.group(
                            "running_average"
                        )
                    )
                    / 1000.0
                ),
                "ram_used_mb": (
                    int(ram_match.group("used"))
                    if ram_match is not None
                    else None
                ),
                "ram_total_mb": (
                    int(ram_match.group("total"))
                    if ram_match is not None
                    else None
                ),
                "gpu_temperature_c": (
                    float(
                        temperature_match.group(
                            "temperature"
                        )
                    )
                    if temperature_match is not None
                    else None
                ),
                "gr3d_utilization_percent": (
                    int(
                        gr3d_match.group(
                            "utilization"
                        )
                    )
                    if gr3d_match is not None
                    else None
                ),
            }
            samples.append(sample)

    if not samples:
        raise RuntimeError(
            f"No valid tegrastats samples found in {path}"
        )

    return samples


def numeric_summary(values):
    values = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
    ]

    if not values:
        return None

    ordered = sorted(values)
    count = len(ordered)

    if count == 1:
        median = ordered[0]
    elif count % 2 == 1:
        median = ordered[count // 2]
    else:
        median = (
            ordered[count // 2 - 1]
            + ordered[count // 2]
        ) / 2.0

    return {
        "samples": count,
        "mean": statistics.fmean(ordered),
        "sample_std": (
            statistics.stdev(ordered)
            if count > 1
            else 0.0
        ),
        "min": ordered[0],
        "median": median,
        "max": ordered[-1],
    }


def discard_boundary_seconds(samples):
    first_second = samples[0]["timestamp"]
    last_second = samples[-1]["timestamp"]

    trimmed = [
        sample
        for sample in samples
        if sample["timestamp"] != first_second
        and sample["timestamp"] != last_second
    ]

    if len(trimmed) < 10:
        raise RuntimeError(
            "Too few idle samples after discarding "
            "boundary seconds"
        )

    return trimmed


def ceil_to_next_second(value):
    without_microseconds = value.replace(
        microsecond=0
    )

    if value.microsecond == 0:
        return without_microseconds

    return (
        without_microseconds
        + datetime.timedelta(seconds=1)
    )


for required_path in (
    BENCHMARK_JSON,
    IDLE_LOG,
    ACTIVE_LOG,
):
    require_file(required_path)

with BENCHMARK_JSON.open(
    "r",
    encoding="utf-8",
) as file:
    benchmark = json.load(file)

window = benchmark["active_window"]
start_time = datetime.datetime.fromisoformat(
    window["start_iso"]
)
end_time = datetime.datetime.fromisoformat(
    window["end_iso"]
)
duration_s = float(window["duration_s"])
inference_count = int(window["inference_count"])

if end_time <= start_time:
    raise RuntimeError(
        "Benchmark active window is invalid"
    )

if duration_s <= 0 or inference_count <= 0:
    raise RuntimeError(
        "Benchmark duration or inference count is invalid"
    )

idle_all = parse_log(IDLE_LOG)
active_all = parse_log(ACTIVE_LOG)

idle_samples = discard_boundary_seconds(idle_all)

# tegrastats timestamps have one-second resolution. Exclude the
# partial start and end seconds so only complete seconds inside the
# measured benchmark window are analyzed.
active_start_inclusive = ceil_to_next_second(
    start_time
)
active_end_exclusive = end_time.replace(
    microsecond=0
)

active_samples = [
    sample
    for sample in active_all
    if (
        sample["timestamp"]
        >= active_start_inclusive
        and sample["timestamp"]
        < active_end_exclusive
    )
]

if len(idle_samples) < 100:
    raise RuntimeError(
        f"Too few idle samples: {len(idle_samples)}"
    )

if len(active_samples) < 100:
    raise RuntimeError(
        "Too few active-window samples: "
        f"{len(active_samples)}"
    )

idle_power = numeric_summary(
    sample["vdd_in_w"]
    for sample in idle_samples
)
active_power = numeric_summary(
    sample["vdd_in_w"]
    for sample in active_samples
)
active_ram = numeric_summary(
    sample["ram_used_mb"]
    for sample in active_samples
)
active_gpu_temperature = numeric_summary(
    sample["gpu_temperature_c"]
    for sample in active_samples
)
active_gr3d = numeric_summary(
    sample["gr3d_utilization_percent"]
    for sample in active_samples
)

idle_mean_w = idle_power["mean"]
active_mean_w = active_power["mean"]
incremental_power_w = active_mean_w - idle_mean_w

if incremental_power_w <= 0:
    raise RuntimeError(
        "Incremental power is not positive. "
        "Check log alignment."
    )

gross_total_energy_j = (
    active_mean_w * duration_s
)
idle_equivalent_energy_j = (
    idle_mean_w * duration_s
)
incremental_total_energy_j = (
    incremental_power_w * duration_s
)

gross_energy_per_inference_j = (
    gross_total_energy_j / inference_count
)
incremental_energy_per_inference_j = (
    incremental_total_energy_j
    / inference_count
)
throughput_fps = float(
    benchmark["timing_ms"]["throughput_fps"]
)

result = {
    "summary_version": "1.0",
    "model": benchmark["model"],
    "input": benchmark["input"],
    "power_mode": "15W",
    "power_rail": "VDD_IN",
    "benchmark_file": str(BENCHMARK_JSON),
    "idle_log": str(IDLE_LOG),
    "active_log": str(ACTIVE_LOG),
    "active_window": {
        "benchmark_start_iso": (
            window["start_iso"]
        ),
        "benchmark_end_iso": window["end_iso"],
        "analyzed_start_inclusive": (
            active_start_inclusive.isoformat()
        ),
        "analyzed_end_exclusive": (
            active_end_exclusive.isoformat()
        ),
        "benchmark_duration_s": duration_s,
        "inference_count": inference_count,
    },
    "idle_power_w": idle_power,
    "active_power_w": active_power,
    "incremental_power_w": incremental_power_w,
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
            1.0
            / gross_energy_per_inference_j
        ),
        "incremental_inferences_per_joule": (
            1.0
            / incremental_energy_per_inference_j
        ),
        "fps_per_active_watt": (
            throughput_fps / active_mean_w
        ),
    },
    "performance": {
        "timing_ms": benchmark["timing_ms"],
        "throughput_fps": throughput_fps,
    },
    "tegrastats_active_window": {
        "ram_used_mb": active_ram,
        "gpu_temperature_c": (
            active_gpu_temperature
        ),
        "gr3d_utilization_percent": active_gr3d,
    },
    "process_memory": benchmark["memory"],
    "method": {
        "tegrastats_interval_ms": 100,
        "instantaneous_vdd_in_used": True,
        "reported_running_average_not_used": True,
        "idle_boundary_seconds_discarded": True,
        "active_boundary_seconds_discarded": True,
        "same_clean_boot_measurement": True,
        "model_loading_in_power_window": False,
        "gpu_transfer_in_power_window": False,
        "warmup_in_power_window": False,
        "image_preprocessing_in_power_window": False,
        "mask_postprocessing_in_power_window": False,
        "camera_capture_included": False,
        "visualization_included": False,
    },
}

OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_JSON.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        result,
        file,
        indent=2,
    )

print("=" * 72)
print("SAM3 BF16 336px power analysis completed")
print("=" * 72)
print(
    f"Idle samples: {idle_power['samples']}"
)
print(
    f"Active samples: {active_power['samples']}"
)
print(
    f"Idle power: {idle_mean_w:.3f} W"
)
print(
    f"Active power: {active_mean_w:.3f} W"
)
print(
    f"Incremental power: "
    f"{incremental_power_w:.3f} W"
)
print(
    f"Gross energy/inference: "
    f"{gross_energy_per_inference_j:.3f} J"
)
print(
    f"Incremental energy/inference: "
    f"{incremental_energy_per_inference_j:.3f} J"
)
print(
    f"FPS/active watt: "
    f"{throughput_fps / active_mean_w:.4f}"
)
print(f"Saved: {OUTPUT_JSON}")
