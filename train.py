import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as torch_optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import ImageFile

from utils import *
from generator import MDMGenerator
from image_transformer import rotation
from prompt_learner import Conditioner

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_generator(args, device):
    """Return (netG, scale_size, img_size) according to dataset / model_type."""
    dataset = args.dataset.lower()
    mt = args.model_type.lower()
    k_for_gen = args.k if args.k and args.k > 0 else None

    if dataset == "cifar10":
        scale_size, img_size = 32, 32
        netG = MDMGenerator(nz=args.nz, k=k_for_gen, device=device).to(device)
    elif dataset == "imagenet":
        if mt == "incv3":
            scale_size, img_size = 300, 299
            netG = MDMGenerator(inception=True, nz=args.nz, k=k_for_gen, device=device).to(device)
        else:
            scale_size, img_size = 256, 224
            netG = MDMGenerator(nz=args.nz, k=k_for_gen, device=device).to(device)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return netG, scale_size, img_size


def build_surrogate(args, device):
    """Load and freeze the surrogate classifier."""
    dataset = args.dataset.lower()
    if dataset == "cifar10":
        model = load_cifar10_model(args.model_type, device)
    else:
        model = load_model(args.model_type)
        model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def normalize(imgs, dataset: str):
    """Apply the correct normalisation for the active dataset."""
    if dataset == "cifar10":
        return normalize_cifar10(imgs)
    return normalize_imagenet(imgs)



def main():
    parser = argparse.ArgumentParser(
        description="ICGN")
    # ── Dataset ──────────────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="imagenet",choices=["cifar10", "imagenet"], help="Which dataset to train on")
    parser.add_argument("--train_dir", type=str,default="E:/mahabub/ImageNet/ILSVRC2012_img_train", help="Root directory of the training images")

    # ── Training hyper-parameters ─────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 128 for CIFAR-10, 8 for ImageNet)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_subset_size", type=int, default=325000, help="Subset size (default: 50 000 for CIFAR-10, 325 000 for ImageNet)")

    # ── Attack ────────────────────────────────────────────────────────────────
    parser.add_argument("--eps", type=int, default=16,help="L-inf budget in pixel values (divided by 255 internally)")
    parser.add_argument("--label_flag", type=str, default="N8", help="Class subset: CIFAR-10→ ALL  ImageNet→N8/C50")

    # ── Surrogate model ───────────────────────────────────────────────────────
    parser.add_argument("--model_type", type=str, default="res50", help="CIFAR-10: cifar10_resnet56 / cifar10_vgg19_bn" "ImageNet: res50 / incv3")

    # ── Generator ─────────────────────────────────────────────────────────────
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--k", type=int, default=None, help="Gaussian lowpass k (default: 2 for CIFAR-10, 4 for ImageNet; <=0 disables)")

    # ── Conditioner (CLIP-CoCoOp) ─────────────────────────────────────────────
    parser.add_argument("--clip_backbone", type=str, default="ViT-B/16")
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--ctx_init", type=str, default='a photo of a')

    # ── Checkpointing / resume ────────────────────────────────────────────────
    parser.add_argument("--save_dir", type=str, default=None, help="Checkpoint root (default: checkpoints_cifar10 or checkpoints_imagenet)")
    parser.add_argument("--load_g_path", type=str, default=None, help="Optional path to a pre-trained generator checkpoint")
    parser.add_argument("--load_cond_path", type=str, default=None, help="Optional path to a pre-trained conditioner checkpoint")
    args = parser.parse_args()

    dataset = args.dataset.lower()

    if args.k is None:
        args.k = 2 if dataset == "cifar10" else 4

    if args.save_dir is None:
        args.save_dir = f"checkpoints_{dataset}"

    print(args)

    # ── Seeding & device ──────────────────────────────────────────────────────
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eps = args.eps / 255.0

    # ── Checkpoint directory ──────────────────────────────────────────────────
    save_dir = os.path.join(args.save_dir, args.model_type)
    os.makedirs(save_dir, exist_ok=True)

    # ── Build generator ───────────────────────────────────────────────────────
    netG, scale_size, img_size = build_generator(args, device)
    if args.load_g_path and os.path.isfile(args.load_g_path):
        print(f"[TRAIN] Loading generator from: {args.load_g_path}")
        netG.load_state_dict(torch.load(args.load_g_path, map_location=device))

    # ── Surrogate model (frozen) ──────────────────────────────────────────────
    model = build_surrogate(args, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_set = get_data(args.train_dir, scale_size, img_size, subset_size=args.train_subset_size,seed=args.seed,)
    print(f"Training data size: {len(train_set)}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # ── Class subset & classnames ─────────────────────────────────────────────
    label_set = get_classes(args.label_flag, dataset=dataset)
    class_ids = np.array(sorted(np.array(label_set, dtype=np.int64).tolist()), dtype=np.int64)
    K = len(class_ids)
    print(f"Target subset (global ids) K={K}: {class_ids.tolist()}")

    class_index = getClassIndex(dataset)
    classnames = [class_index[int(gid)][1].replace("_", " ") for gid in class_ids]

    print("Conditioner classes (local <- global):")
    for local, gid in enumerate(class_ids):
        print(f"  local {local:3d} <- global {int(gid):4d} : '{classnames[local]}'")

    # ── Conditioner ───────────────────────────────────────────────────────────
    prompt_learner = Conditioner(classnames=classnames, clip_backbone=args.clip_backbone, device=device, n_ctx=args.n_ctx, ctx_init=args.ctx_init,).to(device)

    if args.load_cond_path and os.path.isfile(args.load_cond_path):
        print(f"[TRAIN] Loading conditioner from: {args.load_cond_path}")
        prompt_learner.load_state_dict(torch.load(args.load_cond_path, map_location=device))

    # ── Global→Local LUT ──────────────────────────────────────────────────────
    lut_size = 10 if dataset == "cifar10" else 1000
    map_global_to_local = torch.full((lut_size,), -1, device=device, dtype=torch.long)
    map_global_to_local[torch.as_tensor(class_ids, device=device, dtype=torch.long)] = torch.arange(K, device=device, dtype=torch.long)

    # ── Objective + optimizer ─────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    params = list(netG.parameters()) + list(prompt_learner.parameters())
    optimizer = torch_optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.start_epoch, args.epochs):
        netG.train()
        prompt_learner.train()

        running_loss = 0.0
        running_n = 0

        for step, (imgs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            img     = imgs[0].to(device, non_blocking=True)
            img_aug = imgs[1].to(device, non_blocking=True)
            img_rot = rotation(img)[0]

            label_np = np.random.choice(class_ids, size=img.size(0), replace=True)
            label = torch.from_numpy(label_np).to(device=device, dtype=torch.long)
            local_idx = map_global_to_local[label]
            optimizer.zero_grad(set_to_none=True)

            cond = prompt_learner(img, local_idx)

            adv     = netG(input=img,     cond=cond, eps=eps).clamp(0.0, 1.0)
            adv_rot = netG(input=img_rot, cond=cond, eps=eps).clamp(0.0, 1.0)
            adv_aug = netG(input=img_aug, cond=cond, eps=eps).clamp(0.0, 1.0)

            logits     = model(normalize(adv,     dataset))
            logits_rot = model(normalize(adv_rot, dataset))
            logits_aug = model(normalize(adv_aug, dataset))

            loss = (criterion(logits,     label) + criterion(logits_rot, label) + criterion(logits_aug, label))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_n    += 1

            if (step + 1) % 100 == 0:
                avg = running_loss / running_n
                print(f"Epoch {epoch} | Step {step+1}/{len(train_loader)} | loss={avg:.5f}")
                running_loss = 0.0
                running_n    = 0

        # Save epoch checkpoints
        torch.save(netG.state_dict(),           os.path.join(save_dir, f"model-{epoch}.pth"))
        torch.save(prompt_learner.state_dict(), os.path.join(save_dir, f"prompt-{epoch}.pth"))
        print(f"[TRAIN] Saved model-{epoch}.pth and prompt-{epoch}.pth → {save_dir}")

    print("Training complete.")


if __name__ == "__main__":
    main()
