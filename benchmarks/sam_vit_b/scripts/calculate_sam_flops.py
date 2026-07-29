import json
from pathlib import Path

import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis, flop_count_table

from segment_anything import sam_model_registry


CHECKPOINT = Path(
    "checkpoints/sam_vit_b_01ec64.pth"
)
OUTPUT = Path(
    "benchmark_results/sam_vit_b_flops.json"
)


class SAMOnePointForward(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam

    def forward(
        self,
        image,
        point_coords,
        point_labels,
    ):
        image_embeddings = self.sam.image_encoder(
            image
        )

        sparse_embeddings, dense_embeddings = (
            self.sam.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
        )

        low_res_masks, iou_predictions = (
            self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=(
                    self.sam.prompt_encoder
                    .get_dense_pe()
                ),
                sparse_prompt_embeddings=(
                    sparse_embeddings
                ),
                dense_prompt_embeddings=(
                    dense_embeddings
                ),
                multimask_output=True,
            )
        )

        return low_res_masks, iou_predictions


def main():
    OUTPUT.parent.mkdir(exist_ok=True)

    sam = sam_model_registry["vit_b"](
        checkpoint=str(CHECKPOINT)
    )
    sam.eval()
    sam.cuda()

    wrapper = SAMOnePointForward(sam)
    wrapper.eval()

    # FLOPs do not depend on FP16 versus FP32 arithmetic width.
    # FP32 is used here only to simplify tracing.
    image = torch.randn(
        1, 3, 1024, 1024,
        device="cuda",
        dtype=torch.float32,
    )

    point_coords = torch.tensor(
        [[[512.0, 512.0]]],
        device="cuda",
        dtype=torch.float32,
    )

    point_labels = torch.tensor(
        [[1]],
        device="cuda",
        dtype=torch.int64,
    )

    with torch.inference_mode():
        analysis = FlopCountAnalysis(
            wrapper,
            (
                image,
                point_coords,
                point_labels,
            ),
        )

        total_flops = int(analysis.total())
        by_module = {
            name: int(value)
            for name, value
            in analysis.by_module().items()
        }
        unsupported = {
            str(name): int(count)
            for name, count
            in analysis.unsupported_ops().items()
        }

    result = {
        "model": "SAM ViT-B",
        "input": [1, 3, 1024, 1024],
        "prompt": "one positive point",
        "multimask_output": True,
        "tool": "fvcore FlopCountAnalysis",
        "counting_convention": (
            "fvcore counts one fused "
            "multiply-add as one FLOP"
        ),
        "total_flops": total_flops,
        "total_gflops": total_flops / 1e9,
        "by_module_flops": by_module,
        "unsupported_operations": unsupported,
        "warning": (
            "Unsupported operators are not included "
            "in the reported total."
        ),
    }

    OUTPUT.write_text(json.dumps(result, indent=2))

    print("========================================")
    print("SAM ViT-B FLOPs")
    print("========================================")
    print(
        f"Total: {total_flops:,} operations"
    )
    print(
        f"Total: {total_flops / 1e9:.3f} GFLOPs"
    )
    print("Unsupported:", unsupported)
    print()
    print(flop_count_table(analysis, max_depth=2))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
