import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img
import torch.nn.functional as F


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs


class PairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir)
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'input')))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		source_img = read_img(os.path.join(self.root_dir, 'input', img_name)) * 2 - 1

		# print('1', source_img.shape)

		target_img = read_img(os.path.join(self.root_dir, 'target', img_name)) * 2 - 1

		# print('2', target_img.shape)

		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)


			# print('3', source_img.shape)
			# print('4', target_img.shape)

			# 确保图像尺寸一致
			assert source_img.shape == target_img.shape, f"Image shapes do not match: {source_img.shape} != {target_img.shape}"

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}

class PairLoaderForTrain(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir)
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'input')))
		self.img_num = len(self.img_names)

		# 定义期望形状
		self.expected_shapes = {
			'source': (3, self.size, self.size),
			'mask': (3, self.size * 2, self.size * 2),
			'target': (3, self.size, self.size),
		}

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		source_img = read_img(os.path.join(self.root_dir, 'input', img_name)) * 2 - 1

		# print('1', source_img.shape)

		target_img = read_img(os.path.join(self.root_dir, 'target', img_name)) * 2 - 1
		mask_img = read_img(os.path.join(self.root_dir, 'mask', img_name)) * 2 - 1

		# print('2', target_img.shape)

		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)

			# print('3', source_img.shape)
			# print('4', target_img.shape)

		# 转换为 (C, H, W) 格式
		source_tensor = torch.tensor(hwc_to_chw(source_img))
		mask_tensor = torch.tensor(hwc_to_chw(mask_img))
		target_tensor = torch.tensor(hwc_to_chw(target_img))

		# 检查并调整形状
		source_tensor = self._check_and_resize(source_tensor, "source")
		mask_tensor = self._check_and_resize(mask_tensor, "mask")
		target_tensor = self._check_and_resize(target_tensor, "target")

		return {'source': source_tensor, 'mask': mask_tensor, 'target': target_tensor, 'filename': img_name}

	# {'source': hwc_to_chw(source_img), 'mask': hwc_to_chw(mask_img), 'target': hwc_to_chw(target_img), 'filename': img_name}

	def _check_and_resize(self, tensor, key):
		"""
        检查张量形状，如果不符合预期则调整形状。
        """
		expected_shape = self.expected_shapes[key]
		if tensor.shape != expected_shape:
			# print(
			# 	f"Shape inconsistency detected for key '{key}': Expected {expected_shape}, but got {tensor.shape}. Resizing...")
			# 调整形状
			tensor = F.interpolate(tensor.unsqueeze(0), size=expected_shape[1:], mode='bilinear',
								   align_corners=False).squeeze(0)
		return tensor


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}