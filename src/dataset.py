"""Dataset and image augmentation utilities for rice-leaf classification.

Expected layout (one directory per class)::

	data/
		Bacterial_leaf_blight/
		Brown_spot/
		Leaf_smut/
"""

from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RiceLeafDataset(Dataset):

	def __init__(
		self,
		root: str,
		transform: Optional[Callable] = None,
		class_to_idx: Optional[dict] = None,
	):
		self.root = Path(root).expanduser()
		if not self.root.is_dir():
			raise FileNotFoundError(f"Dataset directory not found: {self.root}")

		class_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
		if not class_dirs:
			raise ValueError(f"No class subdirectories found in {self.root}")

		self.classes = [path.name for path in class_dirs]
		self.class_to_idx = class_to_idx or {
			class_name: index for index, class_name in enumerate(self.classes)
		}
		missing_classes = set(self.classes) - set(self.class_to_idx)
		if missing_classes:
			raise ValueError(f"class_to_idx is missing classes: {sorted(missing_classes)}")

		self.samples = []
		for class_dir in class_dirs:
			label = self.class_to_idx[class_dir.name]
			for path in sorted(class_dir.rglob("*")):
				if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
					self.samples.append((path, label))

		if not self.samples:
			raise ValueError(f"No supported image files found in {self.root}")
		self.transform = transform

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int):
		image_path, label = self.samples[index]
		with Image.open(image_path) as image:
			image = image.convert("RGB")
		if self.transform is not None:
			image = self.transform(image)
		return image, label


def get_transforms(image_size: int = 224, train: bool = True):
	"""Return ImageNet-normalized transforms with augmentation for training."""
	normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
	if train:
		return transforms.Compose(
			[
				transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
				transforms.RandomHorizontalFlip(),
				transforms.RandomVerticalFlip(p=0.2),
				transforms.RandomRotation(15),
				transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
				transforms.ToTensor(),
				normalize,
			]
		)
	return transforms.Compose(
		[
			transforms.Resize(int(image_size * 1.14)),
			transforms.CenterCrop(image_size),
			transforms.ToTensor(),
			normalize,
		]
	)
