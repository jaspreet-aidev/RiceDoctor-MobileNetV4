"""Colab training script for an ImageFolder dataset.

Dataset layout:
	/content/drive/MyDrive/RiceDoctor/data/
		train/<class_name>/*.jpg
		val/<class_name>/*.jpg

In Colab, mount Drive, install timm with `!pip install timm`, then run:
	!python /path/to/trian.py --data-dir /content/drive/MyDrive/RiceDoctor/data
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data-dir", type=Path, default=Path("/content/drive/MyDrive/RiceDoctor/data"))
	parser.add_argument("--output-dir", type=Path, default=Path("/content/drive/MyDrive/RiceDoctor/checkpoints"))
	parser.add_argument("--model", default="mobilenetv4_conv_small.e2400_r224_in1k")
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--image-size", type=int, default=224)
	parser.add_argument("--workers", type=int, default=2)
	args = parser.parse_args()

	train_dir, val_dir = args.data_dir / "train", args.data_dir / "val"
	if not train_dir.is_dir() or not val_dir.is_dir():
		raise FileNotFoundError(f"Expected train/ and val/ folders inside {args.data_dir}")

	try:
		import timm
	except ImportError as exc:
		raise RuntimeError("Install the model dependency first: !pip install timm") from exc

	size = args.image_size
	normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	train_data = datasets.ImageFolder(train_dir, transform=transforms.Compose([
		transforms.RandomResizedCrop(size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	]))
	val_data = datasets.ImageFolder(val_dir, transform=transforms.Compose([
		transforms.Resize((size, size)),
		transforms.ToTensor(),
		normalize,
	]))
	if train_data.classes != val_data.classes:
		raise ValueError("train/ and val/ must contain the same class folders")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	use_cuda = device.type == "cuda"
	train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
							  num_workers=args.workers, pin_memory=use_cuda)
	val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
							num_workers=args.workers, pin_memory=use_cuda)

	model = timm.create_model(args.model, pretrained=True, num_classes=len(train_data.classes)).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
	scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	best_accuracy = 0.0
	for epoch in range(args.epochs):
		model.train()
		train_total_loss = 0.0
		for images, labels in train_loader:
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad(set_to_none=True)
			with torch.cuda.amp.autocast(enabled=use_cuda):
				loss = criterion(model(images), labels)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			train_total_loss += loss.item() * images.size(0)

		model.eval()
		val_total_loss, correct = 0.0, 0
		with torch.no_grad():
			for images, labels in val_loader:
				images, labels = images.to(device), labels.to(device)
				with torch.cuda.amp.autocast(enabled=use_cuda):
					logits = model(images)
					loss = criterion(logits, labels)
				val_total_loss += loss.item() * images.size(0)
				correct += (logits.argmax(1) == labels).sum().item()

		accuracy = correct / len(val_data)
		print(f"Epoch {epoch + 1}/{args.epochs} - train loss: {train_total_loss / len(train_data):.4f} - "
			  f"val loss: {val_total_loss / len(val_data):.4f} - val accuracy: {accuracy:.2%}")
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			torch.save({
				"model_state_dict": model.state_dict(),
				"classes": train_data.classes,
				"model_name": args.model,
				"image_size": size,
				"val_accuracy": accuracy,
			}, args.output_dir / "best_model.pth")

	print(f"Best validation accuracy: {best_accuracy:.2%}")
	print(f"Saved checkpoint to {args.output_dir / 'best_model.pth'}")


if __name__ == "__main__":
	main()
