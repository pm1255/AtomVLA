"""CPU contract checks for Query, RGB-pixel supervision, pose, and Pi-style MoT actions."""
import json
import random
import time
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from full_ft.model import FullAtomVLA
from full_ft.data import REGISTRY
from full_ft.schedule import gates


ARCH = {
    "queries": 4,
    "horizon": 8,
    "expert_hidden": 32,
    "expert_layers": 1,
    "expert_heads": 4,
    "pixel_views": 3,
    "pixel_hidden": 32,
    "pixel_layers": 1,
    "pixel_heads": 4,
    "pixel_grid_size": 4,
    "pixel_output_size": 8,
    "gradient_checkpointing": True,
    "sampling_steps": 2,
}


def tiny_model():
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 1024,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    qwen = Qwen3VLForConditionalGeneration(config)
    tok = SimpleNamespace(
        padding_side="right", pad_token_id=127, eos_token_id=126,
        bos_token_id=125, all_special_ids=[120, 121, 122, 123, 125, 126, 127]
    )
    tok.convert_ids_to_tokens = lambda x: "!"
    proc = SimpleNamespace(tokenizer=tok)
    with patch(
        "vendor_delta.qwen_vl_backbone.Qwen3VLForConditionalGeneration.from_pretrained",
        return_value=qwen,
    ), patch("vendor_delta.qwen_vl_backbone.AutoProcessor.from_pretrained", return_value=proc):
        return FullAtomVLA("/nonexistent/tiny_qwen", REGISTRY, ARCH)


def batch(stage=2, robots=("droid", "libero")):
    n = len(robots)
    ids = torch.tensor([[10, 122, 120, 123, 11, 12, 13, 14]] * n)
    mask = torch.ones_like(ids)
    vl = {
        "input_ids": ids,
        "attention_mask": mask,
        "pixel_values": torch.randn(n * 4, 12),
        "image_grid_thw": torch.tensor([[1, 2, 2]] * n),
    }
    if stage == 1:
        labels = ids.clone(); labels[:, :6] = -100; vl["labels"] = labels
    action = torch.randn(n, 8, 8)
    action_mask = torch.ones(n, 8, 8, dtype=torch.bool)
    for i, robot in enumerate(robots):
        action_mask[i, :, REGISTRY[robot]["action_dim"]:] = False
        action[i, :, REGISTRY[robot]["action_dim"]:] = 0
    pose = torch.randn(n, 2, 8, 9) * 0.01
    pose[..., 3:] = torch.tensor([1., 0., 0., 0., 1., 0.])
    return {
        "vl": vl,
        "state": torch.randn(n, 9),
        "action": action,
        "action_mask": action_mask,
        "robots": list(robots),
        "pose": pose,
        "pose_mask": torch.ones(n, 2, 8, dtype=torch.bool),
        "pixel_target": torch.rand(n, 3, 3, 224, 224),
        "pixel_mask": torch.ones(n, 3, 1, 224, 224, dtype=torch.bool),
    }


def main():
    torch.set_num_threads(2); torch.manual_seed(101); random.seed(101)
    table = Counter((gates(42, 2, step, 0)["pixel"], gates(42, 2, step, 0)["pose"]) for step in range(12000))
    assert all(abs(v / 12000 - .25) < .025 for v in table.values()) and len(table) == 4
    assert abs(sum(not gates(42, 1, i, 0)["query"] for i in range(12000)) / 12000 - .2) < .02
    assert all(gates(42, 2, 0, 0, evaluate=True).values())

    model = tiny_model(); model.configure_stage(2)
    assert all(p.requires_grad for p in model.backbone.parameters())
    calls = {"lm": 0, "pixel": 0, "pose": 0}
    def hook(name):
        def called(*args): calls[name] += 1
        return called
    handles = [
        model.backbone.model.lm_head.register_forward_hook(hook("lm")),
        model.pixel.register_forward_hook(hook("pixel")),
        model.pose.register_forward_hook(hook("pose")),
    ]
    b = batch(); model.train()
    loss, metrics = model(b, 2, {"query": True, "pixel": False, "pose": False}, {"pixel": .1, "pose": .1})
    loss.backward()
    assert calls == {"lm": 0, "pixel": 0, "pose": 0}
    assert model.backbone.difference_query.weight.grad.norm() > 0
    assert all(p.grad is None for p in model.pixel.parameters())
    assert all(p.grad is None for p in model.pose.parameters())
    for pixel, pose in [(True, False), (False, True), (True, True)]:
        model.zero_grad(set_to_none=True); calls.update(lm=0, pixel=0, pose=0)
        loss, metrics = model(b, 2, {"query": True, "pixel": pixel, "pose": pose}, {"pixel": .1, "pose": .1})
        loss.backward()
        assert calls == {"lm": 0, "pixel": int(pixel), "pose": int(pose)}
        assert ("pixel_loss" in metrics) == pixel and ("pose_loss" in metrics) == pose
    for handle in handles: handle.remove()

    model.configure_stage(1); bs = batch(1); model.zero_grad(set_to_none=True)
    loss, _ = model(bs, 1, {"query": False, "pixel": False, "pose": False}, {"pixel": 1., "pose": .25})
    loss.backward(); assert model.backbone.difference_query.weight.grad is None
    model.configure_stage(2); model.eval()
    q = model.encode(b["vl"], True, False)["backbone_embeddings"]
    sampled = model.action.sample(q, b["state"], b["robots"], 2)
    assert sampled.shape == (2, 8, 8) and torch.equal(sampled[1, :, 7], torch.zeros(8))
    print(json.dumps({"passed": True, "device": "CPU", "model": "small randomly initialized Qwen3-VL", "mot_prefix_suffix": True, "pixel_auxiliary_supervision": True, "time": time.time()}))


if __name__ == "__main__":
    main()
