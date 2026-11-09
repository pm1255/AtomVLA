"""Full-parameter AtomVLA: continuous Query, RGB-pixel/pose heads, and Pi-style MoT actions.

The VLM produces continuous Difference Query states from the current images and
instruction. Auxiliary heads read those states during representation learning;
the action expert is a separate Flow-Matching network whose action tokens are
concatenated *after* the Query prefix, as in Pi-style prefix/suffix training.
There is deliberately no optical-flow prediction or optical-flow loss here.
"""
import torch
from torch import nn
from torch.nn import functional as F

from vendor_delta.qwen_vl_backbone import QwenVLBackbone


class PixelDecoderLayer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)
        )
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, grid, queries):
        attended = self.attention(grid, queries, queries, need_weights=False)[0]
        grid = self.norm(grid + attended)
        return self.output_norm(grid + self.mlp(grid))


class PixelDecoder(nn.Module):
    """Decode Query states into a low-resolution future RGB image per camera."""

    def __init__(self, input_dim, num_queries, num_views=3, hidden=256,
                 layers=2, heads=8, grid_size=14, output_size=56):
        super().__init__()
        if num_views <= 0 or grid_size <= 0 or output_size <= 0:
            raise ValueError("pixel decoder dimensions must be positive")
        self.num_queries = num_queries
        self.num_views = num_views
        self.output_size = output_size
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden)
        self.grid_size = grid_size
        self.grid = nn.Parameter(torch.empty(1, grid_size * grid_size, hidden))
        nn.init.normal_(self.grid, std=0.02)
        self.layers = nn.ModuleList(
            [PixelDecoderLayer(hidden, heads) for _ in range(layers)]
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.output = nn.Conv2d(hidden, 3 * num_views, 3, padding=1)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, query_states):
        if query_states.ndim != 3 or query_states.shape[1] < self.num_queries:
            raise ValueError("pixel decoder requires [B,Nq,H] with Nq >= num_queries")
        queries = query_states[:, -self.num_queries:, :]
        queries = self.input_projection(
            self.input_norm(queries).to(self.input_projection.weight.dtype)
        )
        grid = self.grid.expand(queries.shape[0], -1, -1)
        for layer in self.layers:
            grid = layer(grid, queries)
        grid = grid.transpose(1, 2).reshape(
            queries.shape[0], -1, self.grid_size, self.grid_size
        )
        grid = F.interpolate(
            grid, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False
        )
        pixels = self.output(self.refinement(grid))
        pixels = pixels.view(
            pixels.shape[0], self.num_views, 3, self.output_size, self.output_size
        )
        return pixels.sigmoid()


class NativeActionExpert(nn.Module):
    """Pi-style MoT action expert with Query prefix and action-token suffix.

    Query tokens are projected to the expert width and concatenated before the
    noisy action tokens. The suffix can attend to the complete Query prefix and
    to previous action tokens, while the prefix cannot attend back into actions.
    """

    def __init__(self, query_dim, registry, horizon=8, hidden=512, layers=8, heads=8):
        super().__init__()
        if hidden % heads:
            raise ValueError("expert_hidden must be divisible by expert_heads")
        self.registry = registry
        self.horizon = horizon
        self.hidden = hidden
        self.max_action = max(v["action_dim"] for v in registry.values())
        self.context = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, hidden))
        self.action_in = nn.ModuleDict(
            {k: nn.Linear(v["action_dim"], hidden) for k, v in registry.items()}
        )
        self.state_in = nn.ModuleDict(
            {k: nn.Linear(v["state_dim"], hidden) for k, v in registry.items()}
        )
        self.action_out = nn.ModuleDict(
            {k: nn.Linear(hidden, v["action_dim"]) for k, v in registry.items()}
        )
        self.robot = nn.ParameterDict(
            {k: nn.Parameter(torch.randn(hidden) * 0.02) for k in registry}
        )
        self.position = nn.Parameter(torch.randn(1, horizon, hidden) * 0.02)
        self.time = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerEncoder(layer, layers)

    def _attention_mask(self, prefix_len, device):
        mask = torch.zeros(
            prefix_len + self.horizon,
            prefix_len + self.horizon,
            dtype=torch.bool,
            device=device,
        )
        mask[:prefix_len, prefix_len:] = True
        mask[prefix_len:, prefix_len:] = torch.triu(
            torch.ones(self.horizon, self.horizon, dtype=torch.bool, device=device), 1
        )
        return mask

    def forward(self, q, state, noisy, t, robots):
        if q.ndim != 3 or noisy.ndim != 3 or len(robots) != len(q):
            raise ValueError("invalid Query/action batch shapes")
        context = self.context(q.float())
        outputs = []
        for i, key in enumerate(robots):
            if key not in self.registry:
                raise ValueError(f"Unverified action space: {key}")
            spec = self.registry[key]
            d, sd = spec["action_dim"], spec["state_dim"]
            x = self.action_in[key](noisy[i:i + 1, :, :d])
            x = x + self.state_in[key](state[i:i + 1, :sd])[:, None]
            clock = torch.stack(
                [
                    t[i:i + 1],
                    torch.sin(t[i:i + 1] * torch.pi),
                    torch.cos(t[i:i + 1] * torch.pi),
                ],
                -1,
            )
            x = x + self.position + self.robot[key] + self.time(clock)[:, None]
            prefix = context[i:i + 1].to(x.dtype)
            joint = torch.cat([prefix, x], dim=1)
            encoded = self.decoder(
                joint, mask=self._attention_mask(prefix.shape[1], joint.device)
            )
            y = self.action_out[key](encoded[:, -self.horizon:])
            outputs.append(F.pad(y, (0, self.max_action - d)))
        return torch.cat(outputs)

    def loss(self, q, state, action, mask, robots):
        eps = torch.randn_like(action)
        t = torch.rand(len(action), device=action.device)
        noisy = ((1 - t[:, None, None]) * eps + t[:, None, None] * action) * mask
        velocity = self(q, state, noisy, t, robots)
        per = ((velocity - (action - eps)).square() * mask).sum((1, 2))
        per = per / mask.sum((1, 2)).clamp_min(1)
        if not (mask.sum((1, 2)) > 0).all():
            raise ValueError("Action-loss sample has no valid controls")
        return per.mean()

    @torch.no_grad()
    def sample(self, q, state, robots, steps=10, generator=None):
        a = torch.randn(
            (len(q), self.horizon, self.max_action), device=q.device, generator=generator
        )
        for i, key in enumerate(robots):
            a[i, :, self.registry[key]["action_dim"]:] = 0
        for i in range(steps):
            t = torch.full((len(q),), i / steps, device=q.device)
            a = a + self(q, state, a, t, robots) / steps
        return a


class FullAtomVLA(nn.Module):
    def __init__(self, base, registry, config, backbone=None):
        super().__init__()
        self.config = config
        self.backbone = backbone if backbone is not None else QwenVLBackbone(
            base,
            True,
            {"use_lora": False},
            use_difference_query=True,
            num_difference_queries=config["queries"],
            vlm_attention_backend="sdpa",
        )
        self.backbone.float().requires_grad_(True)
        if any("lora_" in n.lower() for n, _ in self.backbone.named_parameters()):
            raise ValueError("LoRA adapters are forbidden in full-parameter training")
        model = self.backbone.model
        model.config.use_cache = False
        if config.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.backbone.processor.tokenizer.padding_side = "right"
        h = self.backbone.difference_query.hidden_size
        self.pixel = PixelDecoder(
            h,
            config["queries"],
            num_views=config.get("pixel_views", 3),
            hidden=config.get("pixel_hidden", 256),
            layers=config.get("pixel_layers", 2),
            heads=config.get("pixel_heads", 8),
            grid_size=config.get("pixel_grid_size", 14),
            output_size=config.get("pixel_output_size", 56),
        )
        self.pose = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, 256), nn.GELU(), nn.Linear(256, 2 * 8 * 9)
        )
        with torch.no_grad():
            self.pose[-1].bias.reshape(2, 8, 9)[..., 3:] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            )
        self.action = NativeActionExpert(
            h,
            registry,
            config["horizon"],
            config["expert_hidden"],
            config["expert_layers"],
            config["expert_heads"],
        )

    @property
    def processor(self):
        return self.backbone.processor

    def configure_stage(self, stage):
        self.requires_grad_(True)
        if stage == 1:
            self.action.requires_grad_(False)
        if not all(p.requires_grad for p in self.backbone.parameters()):
            raise AssertionError("VLM is not fully unfrozen")

    def encode(self, vl, use_query, coding):
        if use_query:
            return self.backbone(vl, compute_vlm_loss=coding)
        if not coding:
            raise ValueError("Stage 2 always retains Query")
        out = self.backbone.model(**vl, return_dict=True, use_cache=False)
        return {"vlm_loss": out.loss, "backbone_embeddings": None}

    def _pixel_loss(self, prediction, target, mask):
        if target.ndim != 5 or mask.ndim != 5:
            raise ValueError("pixel target/mask must be [B,V,C,H,W]/[B,V,1,H,W]")
        if target.shape[:2] != mask.shape[:2] or target.shape[2] != 3 or mask.shape[2] != 1:
            raise ValueError("pixel target/mask channel or view dimensions are invalid")
        if target.shape[1] > prediction.shape[1]:
            raise ValueError("batch contains more camera views than pixel_views")
        prediction = prediction[:, :target.shape[1]]
        size = prediction.shape[-1]
        target = F.interpolate(
            target.flatten(0, 1), size=(size, size), mode="area"
        ).view_as(prediction)
        valid = F.interpolate(
            mask.float().flatten(0, 1), size=(size, size), mode="area"
        ).view(mask.shape[0], mask.shape[1], 1, size, size)
        if not torch.isfinite(target).all() or not torch.isfinite(valid).all():
            raise ValueError("non-finite RGB target")
        valid = valid.expand_as(prediction)
        residual = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none")
        denom = valid.sum().clamp_min(1.0)
        loss = (residual * valid).sum() / denom
        mae = ((prediction.float() - target.float()).abs() * valid).sum() / denom
        return loss, mae

    def forward(self, batch, stage, gates, weights, evaluate=False):
        use_query = bool(gates["query"])
        if stage == 2 and (not use_query or "labels" in batch["vl"]):
            raise ValueError("Stage 2 requires Query and forbids target text")
        if evaluate and not all(gates.values()):
            raise ValueError("Validation must disable all loss dropout")
        out = self.encode(batch["vl"], use_query, stage == 1)
        metrics = {}
        loss = out["vlm_loss"] if stage == 1 else None
        if stage == 1:
            metrics["code_loss"] = loss.detach()
        if use_query:
            q = out["backbone_embeddings"]
            if stage == 2:
                action_loss = self.action.loss(
                    q, batch["state"], batch["action"], batch["action_mask"], batch["robots"]
                )
                loss = action_loss
                metrics["action_fm_loss"] = action_loss.detach()
            if gates["pixel"]:
                pixel = self.pixel(q)
                pixel_loss, pixel_mae = self._pixel_loss(
                    pixel, batch["pixel_target"], batch["pixel_mask"]
                )
                loss = pixel_loss * weights["pixel"] if loss is None else loss + weights["pixel"] * pixel_loss
                metrics["pixel_loss"] = pixel_loss.detach()
                metrics["pixel_mae"] = pixel_mae.detach()
            if gates["pose"]:
                pose = self.pose(q.float().mean(1)).reshape(-1, 2, 8, 9)
                mask = batch["pose_mask"][..., None]
                pose_loss = (
                    F.smooth_l1_loss(pose, batch["pose"], reduction="none") * mask
                ).sum() / (mask.sum() * 9).clamp_min(1)
                loss = pose_loss * weights["pose"] if loss is None else loss + weights["pose"] * pose_loss
                metrics["pose_loss"] = pose_loss.detach()
                metrics["position_mae_m"] = (
                    ((pose[..., :3] - batch["pose"][..., :3]).abs() * mask).sum()
                    / (mask.sum() * 3).clamp_min(1)
                    * 0.1
                ).detach()
                if evaluate:
                    def matrix(x):
                        a = F.normalize(x[..., :3], dim=-1)
                        b = F.normalize(
                            x[..., 3:] - (a * x[..., 3:]).sum(-1, keepdim=True) * a,
                            dim=-1,
                        )
                        return torch.stack([a, b, torch.linalg.cross(a, b)], -1)

                    rel = matrix(pose[..., 3:]).transpose(-1, -2) @ matrix(
                        batch["pose"][..., 3:]
                    )
                    angle = torch.acos(
                        ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(
                            -1 + 1e-6, 1 - 1e-6
                        )
                    )
                    metrics["rotation_error_deg"] = (
                        (angle * batch["pose_mask"]).sum()
                        / batch["pose_mask"].sum().clamp_min(1)
                        * 180
                        / torch.pi
                    )
            if evaluate and stage == 2:
                sampled = self.action.sample(
                    q,
                    batch["state"],
                    batch["robots"],
                    self.config.get("sampling_steps", 10),
                )
                metrics["action_normalized_mae"] = (
                    (sampled - batch["action"]).abs() * batch["action_mask"]
                ).sum() / batch["action_mask"].sum().clamp_min(1)
        if loss is None:
            raise ValueError("loss is undefined: no active objective")
        metrics["objective"] = loss.detach()
        return loss, metrics
