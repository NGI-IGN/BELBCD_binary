"""
inference_bzone.py

Bi-temporal change-detection inference where the before/after image
directories are passed via CLI args, ChangeFormer-argparse style,
instead of coming from the fixed DATASET_PRESETS dict in config.py.

Usage:
    python inference_bzone.py \
        --before_dir /path/to/epoch_A \
        --after_dir  /path/to/epoch_B \
        --checkpoint ./checkpoints/fotbcd/best.pth \
        --out_dir    ./predictions
"""
import argparse
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from evaluate import load_model  # reuse your existing model loader


def get_args():
    parser = argparse.ArgumentParser(description="Bi-temporal change detection inference")
    parser.add_argument('--before_dir', type=str, required=True,
                         help='Directory of pre-change (epoch A) images')
    parser.add_argument('--after_dir', type=str, required=True,
                         help='Directory of post-change (epoch B) images')
    parser.add_argument('--checkpoint', type=str, required=True,
                         help='Path to model checkpoint (.pth)')
    parser.add_argument('--out_dir', type=str, default='./predictions',
                         help='Directory to save predicted change masks')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5,
                         help='Sigmoid threshold for single-channel logit models')
    parser.add_argument('--ext', type=str, default='.png',
                         help='Image extension used to match before/after pairs')
    return parser.parse_args()


class PairedFolderDataset(Dataset):
    """
    Pairs images by matching filename across two arbitrary folders,
    instead of relying on a pre-registered dataset name / fixed data_root.
    """

    def __init__(self, before_dir, after_dir, img_size, ext='.png'):
        before_names = {f for f in os.listdir(before_dir) if f.lower().endswith(ext)}
        after_names = {f for f in os.listdir(after_dir) if f.lower().endswith(ext)}
        common = sorted(before_names & after_names)

        missing_before = after_names - before_names
        missing_after = before_names - after_names
        if missing_before or missing_after:
            print(f"Warning: {len(missing_before)} file(s) in after_dir with no match "
                  f"in before_dir, {len(missing_after)} file(s) in before_dir with no "
                  f"match in after_dir. These are skipped.")
        if not common:
            raise ValueError("No matching filenames found between before_dir and after_dir.")

        self.before_dir = before_dir
        self.after_dir = after_dir
        self.names = common
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_a = Image.open(os.path.join(self.before_dir, name)).convert('RGB')
        img_b = Image.open(os.path.join(self.after_dir, name)).convert('RGB')
        return {
            'A': self.transform(img_a),
            'B': self.transform(img_b),
            'name': os.path.splitext(name)[0],
        }


def get_loader(args):
    dataset = PairedFolderDataset(
        before_dir=args.before_dir,
        after_dir=args.after_dir,
        img_size=args.img_size,
        ext=args.ext,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )




def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = get_loader(args)
    model = load_model(args.checkpoint, device)

    with torch.no_grad():
        for batch in loader:
            A, B = batch["A"].to(device), batch["B"].to(device)
            out = model(A, B)

            if isinstance(out, (list, tuple)):        # deep supervision -> take last head
                out = out[-1]
            if out.shape[1] == 1:                      # (N,1,H,W) -> sigmoid
                pred = (out.sigmoid()[:, 0] > args.threshold)
            else:                                       # (N,2,H,W) -> argmax
                pred = out.argmax(1)

            pred = pred.to(torch.uint8).cpu().numpy()
            for m, name in zip(pred, batch["name"]):
                Image.fromarray(m * 255).save(os.path.join(args.out_dir, f"{name}.png"))

    print(f"Done. Saved {len(loader.dataset)} prediction(s) to {args.out_dir}")


if __name__ == '__main__':
    main()
[]