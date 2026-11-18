import torch
from pathlib import Path
from config import CFG
from model import build_model
from dataset import get_dataloaders
from losses import build_loss
from train import validate  # reuses the same validate() function

device = CFG.DEVICE
save_dir = Path(CFG.SAVE_DIR) / CFG.EXPERIMENT_NAME

_, _, test_loader = get_dataloaders(
    name=CFG.DATASET, root=CFG.DATA_ROOT, batch_size=CFG.BATCH_SIZE,
    img_size=CFG.IMG_SIZE, num_workers=CFG.NUM_WORKERS,
    crop_size=CFG.CROP_SIZE, original_size=CFG.ORIGINAL_SIZE,
)

model = build_model(CFG).to(device)
model.load_state_dict(torch.load(save_dir / "best_iou.pth", map_location=device))

criterion = build_loss(CFG)
test_m = validate(model, test_loader, criterion, device, desc="Test")
print(f"Test - IoU: {test_m['iou']:.4f}, F1: {test_m['f1']:.4f}, mIoU: {test_m['miou']:.4f}, "
      f"Precision: {test_m['precision']:.4f}, Recall: {test_m['recall']:.4f}")