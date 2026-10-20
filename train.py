import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import os
import json
import hashlib
from utils.utils import seed_everything, configure_mlp, train_model, make_dataset
from torch.utils.data import DataLoader

seed_everything()

X, y = load_digits(return_X_y=True)
idx = np.arange(len(y))
rest, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
train, val = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=0)

os.makedirs("data", exist_ok=True)
split = {k: sorted(map(int, v)) for k, v in
         {"train": train, "val": val, "test": test}.items()}

with open("data/split.json", "w") as f:
    json.dump(split, f, sort_keys=True)

sha = hashlib.sha256(open("data/split.json", "rb").read()).hexdigest()

entry = {
    "source": "sklearn.datasets.load_digits",

    "version": f"scikit-learn {sklearn.__version__}",

    "selection": {"total": len(y), "used": len(y), "excluded": 0},

    "split_rule": ("stratified by class, split_seed=0: test = 20% of all; "
                   "val = 25% of the remainder; train = the rest"),

    "files": [{"path": "data/split.json", "sha256": sha,
               "records": len(y)}],

    "sizes": {"train": len(train), "val": len(val), "test": len(test)},
}
mpath = "data/manifest.json"
manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
manifest["lr0_digits"] = entry
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)


mlp = configure_mlp(n_layers=1, hidden=64, dropout=0.0)

train_ds, val_ds, test_ds = make_dataset(X, y, train), make_dataset(X, y, val), make_dataset(X, y, test)
train_dataloader = DataLoader(train_ds, batch_size=64)
val_dataloader = DataLoader(val_ds, batch_size=64)
test_dataloader = DataLoader(test_ds, batch_size=64)

trained_model = train_model(mlp, num_epochs=30, train_loader=train_dataloader, val_loader=val_dataloader, lr=0.003)