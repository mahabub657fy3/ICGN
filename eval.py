import os
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms

from utils import *
from generator import MDMGenerator
from prompt_learner import Conditioner

def fix_labels(test_set, val_txt_path: str):
    val_dict = {}
    with open(val_txt_path, "r") as f:
        for line in f:
            key, val = line.strip().split(",")
            key = os.path.splitext(os.path.basename(key))[0]
            val_dict[key] = int(val)

    new_samples = []
    for path, _ in test_set.samples:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem not in val_dict:
            raise RuntimeError(f"Filename stem '{stem}' not found in val_txt={val_txt_path}")
        new_samples.append((path, val_dict[stem]))

    test_set.samples = new_samples
    return test_set


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


def main():
    parser = argparse.ArgumentParser(description="ICGN")
    # ── Dataset ───────────────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="imagenet",choices=["cifar10", "imagenet"],help="Which dataset to evaluate on")
    parser.add_argument("--data_dir", type=str, default="E:/mahabub/neurips2017_dev", help="Root directory of the test images")

    # ── Eval settings ─────────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eps", type=int, default=16)
    parser.add_argument("--model_type", type=str, default="incv3", help="CIFAR-10: cifar10_vgg19_bn/ cifar10_resnet56  ImageNet: res50 / incv3")
    parser.add_argument("--label_flag", type=str, default="N8", help="Class subset: CIFAR-10→ALL  ImageNet→N8/C50")

    # ── NIPS/ImageNet label fix ───────────────────────────────────────────────
    parser.add_argument("--is_nips", action="store_true", default=False, help="Use fix_labels_nips (ImageNet NeurIPS dev set)")

    # ── CIFAR-10 label override ───────────────────────────────────────────────
    parser.add_argument("--val_txt", type=str, default='Cifar_10_val.txt', help="Optional CIFAR-10 test label file (e.g. Cifar_10_val.txt)")

    # ── Generator ─────────────────────────────────────────────────────────────
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--k", type=int, default=None, help="Gaussian lowpass k (default: 2 for CIFAR-10, 4 for ImageNet; <=0 disables)")
    parser.add_argument("--load_g_path", type=str, default="model-9.pth")

    # ── Conditioner ───────────────────────────────────────────────────────────
    parser.add_argument("--clip_backbone", type=str, default="ViT-B/16")
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--ctx_init", type=str, default='a photo of a')
    parser.add_argument("--load_cond_path", type=str, default="prompt-9.pth")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--save_dir", type=str, default=None, help="Output root (default: results_cifar10 or results_imagenet)")

    args = parser.parse_args()

    # ── Apply dataset-specific defaults ───────────────────────────────────────
    dataset = args.dataset.lower()

    if args.k is None:
        args.k = 2 if dataset == "cifar10" else 4

    if args.save_dir is None:
        args.save_dir = f"results_{dataset}"

    # ImageNet NeurIPS dev set: enable is_nips by default
    if dataset == "imagenet" and not args.is_nips:
        args.is_nips = True

    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eps = args.eps / 255.0

    # ── Build generator ───────────────────────────────────────────────────────
    netG, scale_size, img_size = build_generator(args, device)
    print(f"[EVAL] Loading generator from: {args.load_g_path}")
    netG.load_state_dict(torch.load(args.load_g_path, map_location=device))
    netG.eval()

    # ── Data ──────────────────────────────────────────────────────────────────
    data_transform = transforms.Compose([transforms.Resize(scale_size), transforms.CenterCrop(img_size), transforms.ToTensor(),])

    test_set = datasets.ImageFolder(args.data_dir, data_transform)

    if dataset == "cifar10" and args.val_txt is not None:
        print(f"[EVAL] Overwriting labels using val_txt: {args.val_txt}")
        test_set = fix_labels(test_set, args.val_txt)

    if dataset == "imagenet" and args.is_nips:
        test_set = fix_labels_nips(args, test_set, pytorch=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,)

    # ── Class subset & classnames ─────────────────────────────────────────────
    class_ids = get_classes(args.label_flag, dataset=dataset)
    used_ids = np.array(sorted(np.array(class_ids, dtype=np.int64).tolist()), dtype=np.int64)
    K = len(used_ids)

    class_index = getClassIndex(dataset)
    classnames = [class_index[int(i)][1].replace("_", " ") for i in used_ids]

    print("Conditioner classes (global -> local):")
    for local, gid in enumerate(used_ids):
        print(f"  global {int(gid):4d} -> local {local:3d} : '{classnames[local]}'")

    # ── Global→Local LUT ──────────────────────────────────────────────────────
    lut_size = 10 if dataset == "cifar10" else 1000
    map_global_to_local = torch.full((lut_size,), -1, device=device, dtype=torch.long)
    map_global_to_local[torch.as_tensor(used_ids, device=device, dtype=torch.long)] = torch.arange(K, device=device, dtype=torch.long)

    # ── Conditioner ───────────────────────────────────────────────────────────
    prompt_learner = Conditioner(classnames=classnames,clip_backbone=args.clip_backbone,device=device,n_ctx=args.n_ctx,ctx_init=args.ctx_init,).to(device)

    print(f"[EVAL] Loading conditioner from: {args.load_cond_path}")
    prompt_learner.load_state_dict(torch.load(args.load_cond_path, map_location=device))
    prompt_learner.eval()

    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    subdir_name = f"gan_{args.label_flag.lower()}"
    to_pil = transforms.ToPILImage(mode="RGB")

    # ── Generate and save ─────────────────────────────────────────────────────
    with torch.no_grad():
        for idx, target_class in enumerate(class_ids):
            target_class = int(target_class)
            print(f"Generating adv examples for target class {target_class} ({idx+1}/{len(class_ids)})")

            target_local = map_global_to_local[torch.tensor(target_class, device=device, dtype=torch.long)].item()
            if target_local < 0:
                raise RuntimeError(f"Target class {target_class} not in subset mapping.")

            output_dir = os.path.join(args.save_dir,subdir_name,f"{args.model_type}_t{target_class}","images",)
            os.makedirs(output_dir, exist_ok=True)

            img_counter = 0
            for img, _ in test_loader:
                img = img.to(device, non_blocking=True)
                b = img.size(0)

                local_idx = torch.full((b,), target_local, dtype=torch.long, device=device)
                cond = prompt_learner(img, local_idx)

                adv = netG(input=img, cond=cond, eps=eps).clamp(0.0, 1.0)

                adv_cpu = adv.cpu()
                for j in range(b):
                    out_path = os.path.join(output_dir, f"{target_class}_{img_counter}.png")
                    to_pil(adv_cpu[j]).save(out_path)
                    img_counter += 1

    print("Done generating adversarial images.")


if __name__ == "__main__":
    main()
