# SPDX-License-Identifier: Apache-2.0
"""Train either stage on one GPU or with torchrun; save weights and resumable state."""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoints import parameters_for_stage, save_weights, weight_state
from .data import RestorationDataset, Stage1Dataset
from .models import build_stage1, build_stage2
from .utils import read_config, to_device


class TrainingLoss(nn.Module):
    """Use DDP's forward path for the model's stage-specific training loss."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.training_loss(batch)


def batches(loader, dataset, sampler, epoch=0, offset=0):
    while True:
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        for index, batch in enumerate(loader):
            if index >= offset:
                yield epoch, index + 1, batch
        epoch, offset = epoch + 1, 0


def train_loop(model, dataset, config, stage, output, resume=None):
    options = config["training"]
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    micro = options["micro_batch_size"]
    if options["batch_size"] % (world * micro):
        raise ValueError("Global batch_size must be divisible by world_size * micro_batch_size")
    accumulation = options["batch_size"] // (world * micro)
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True, seed=config["seed"], drop_last=True
    )
    loader = DataLoader(
        dataset,
        batch_size=micro,
        sampler=sampler,
        drop_last=True,
        num_workers=config["data"]["workers"],
        pin_memory=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    if len(loader) == 0:
        raise ValueError("Training split is too small for the requested batch size")
    parameters = list(parameters_for_stage(model, stage).values())
    optimizer = torch.optim.AdamW(
        parameters, lr=options["learning_rate"], weight_decay=options["weight_decay"]
    )
    wrapper = TrainingLoss(model)
    if world > 1:
        wrapper = DistributedDataParallel(
            wrapper, device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
    step, epoch, offset = 0, 0, 0
    torch.manual_seed(config["seed"] + rank)
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["stage"] != stage or checkpoint["world_size"] != world:
            raise ValueError("Resume requires the same stage and GPU count")
        # Resume files are training artifacts; deployable weights are safetensors.
        if stage == "stage1":
            model.raw_encoder.load_state_dict(checkpoint["weights"])
        else:
            with torch.no_grad():
                for key, p in parameters_for_stage(model, stage).items():
                    p.copy_(checkpoint["weights"][key])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, epoch, offset = (checkpoint[k] for k in ("step", "epoch", "batch_offset"))
        rng = checkpoint["rng_states"][rank]
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"])
    stream = batches(loader, dataset, sampler, epoch, offset)
    model.train()
    log = (output / "train.jsonl").open("a") if rank == 0 else None
    try:
        while step < options["max_steps"]:
            optimizer.zero_grad(set_to_none=True)
            mean_loss = torch.zeros((), device=config["device"])
            for micro_step in range(accumulation):
                epoch, offset, batch = next(stream)
                sync = (
                    wrapper.no_sync()
                    if world > 1 and micro_step < accumulation - 1
                    else nullcontext()
                )
                with sync, torch.autocast("cuda", dtype=torch.bfloat16, enabled=stage == "stage2"):
                    loss, _ = wrapper(to_device(batch, config["device"]))
                    scaled_loss = loss / accumulation
                scaled_loss.backward()
                mean_loss += loss.detach() / accumulation
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, options["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            if world > 1:
                dist.all_reduce(mean_loss)
                mean_loss /= world
            if rank == 0 and (step == 1 or step % options["log_every"] == 0):
                line = json.dumps(
                    {"step": step, "loss": mean_loss.item(), "grad_norm": norm.item()}
                )
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            if step % options["save_every"] == 0 or step == options["max_steps"]:
                rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
                rng_states = [None] * world
                if world > 1:
                    dist.all_gather_object(rng_states, rng)
                else:
                    rng_states[0] = rng
                if rank == 0:
                    save_weights(
                        model,
                        stage,
                        output / f"step{step:06d}.safetensors",
                        config["data"]["crop_size"],
                        step,
                    )
                    state = {k: v.detach().cpu() for k, v in weight_state(model, stage).items()}
                    torch.save(
                        {
                            "stage": stage,
                            "step": step,
                            "epoch": epoch,
                            "batch_offset": offset,
                            "world_size": world,
                            "weights": state,
                            "optimizer": optimizer.state_dict(),
                            "rng_states": rng_states,
                        },
                        output / "last.pt",
                    )
                if world > 1:
                    dist.barrier()
    finally:
        if log:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["stage1", "stage2"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="Resume from this trainer's last.pt")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    local_rank, world = int(os.getenv("LOCAL_RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    config["device"] = f"cuda:{local_rank}"
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device(config["device"]))
    rank = dist.get_rank() if world > 1 else 0
    try:
        torch.manual_seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.max_steps is not None:
            config["training"]["max_steps"] = args.max_steps
        output = Path(args.output)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if any(output.iterdir()) and not args.resume:
                raise FileExistsError(f"Output is not empty: {output}")
            (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        if world > 1:
            dist.barrier()
        data = config["data"]
        if args.stage == "stage1":
            dataset = Stage1Dataset(
                data["manifest"],
                data["crop_size"],
                seed=config["seed"],
                split_key=data.get("split_key", "split"),
            )
            model = build_stage1(config, load_trained=False, perceptual=True)
        else:
            dataset = RestorationDataset(
                data["manifest"],
                data["dataset"],
                data["crop_size"],
                split="train",
                training=True,
                seed=config["seed"],
                include_target=True,
            )
            model = build_stage2(config)
        train_loop(model, dataset, config, args.stage, output, args.resume)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
