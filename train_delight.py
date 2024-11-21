import os
import argparse
import json
import warnings
from torch import nn
from thop import profile, clever_format
from depth.networks import *
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models import DepthNet, DIACMPN, DIACMPN_dehighlight_Indoor
from models.DIACMPN import DetectionLoss
from models.UNet import UNet
from pytorch_ssim import ssim
from utils import AverageMeter
from datasets.loader import PairLoader, PairLoaderForTrain
from loss.CR_loss import ContrastLoss as crloss
import setproctitle
setproctitle.setproctitle("dehighlight")

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Highlight-Removal', type=str, help='dehighlight model name')
parser.add_argument('--model_depth', default='Highlight-Removal-depth', type=str, help='depth model name')
parser.add_argument('--num_workers', default=10, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='RD', type=str, help='dataset name')
parser.add_argument('--exp', default='Highlight-Removal', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0，1，2，3', type=str, help='GPUs used for training')
# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.instancenorm")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def train(train_loader, network, DEPTH_NET, criterion_dehighlight, criterion_dehighlight_cr, criterion_depth,
				   optimizer_dehighlight, optimizer_depth, scaler):
	dehighlight_loss = AverageMeter()
	Detection_loss = AverageMeter()
	torch.cuda.empty_cache()

	network.train()
	DEPTH_NET.train()

	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()
		mask_img = batch['mask'].cuda()

		with autocast(args.no_autocast):
			dehighlight_output_img, d11, d22, d33 = network(source_img)
			dehighlight_output_img_2_depth_img = DEPTH_NET(dehighlight_output_img)

			# real_img_2_depth_img = depth_decoder(encoder(target_img))
			# real_img_2_depth_img = real_img_2_depth_img[("disp", 0)]

			# 使用双线性插值来调整大小
			mask_img = nn.functional.interpolate(mask_img, size=(256, 256), mode='bilinear',
													  align_corners=False)
			# 如果需要，选择特定的通道
			mask_img = mask_img[:, :1, :, :]

			diff_dehighlight = torch.sub(dehighlight_output_img, target_img)
			B, C, H, W = diff_dehighlight.shape
			diff_dehighlight = diff_dehighlight.permute(0, 2, 3, 1)
			diff_dehighlight = diff_dehighlight.reshape(-1, C * H * W)
			epsilon = 1e-7
			diff_d_w = F.softmax(diff_dehighlight, dim=-1) + epsilon
			diff_d_w = diff_d_w.reshape(B, H, W, C).permute(0, 3, 1, 2)
			diff_dehighlight_w = torch.sum(diff_d_w, dim=1, keepdim=True)
			weighted_depth_output_img = torch.mul(dehighlight_output_img_2_depth_img, diff_dehighlight_w)
			# weighted_real_img_2_depth_img = torch.mul(real_img_2_depth_img, diff_dehighlight_w)
			weighted_real_img_2_depth_img = torch.mul(mask_img, diff_dehighlight_w)

			loss_depth_consis = criterion_depth(weighted_depth_output_img, weighted_real_img_2_depth_img)
			# loss_depth_consis_w = criterion_depth(dehighlight_output_img_2_depth_img, real_img_2_depth_img)
			loss_depth_consis_w = criterion_depth(dehighlight_output_img_2_depth_img, mask_img)
			loss_total_depth = loss_depth_consis + loss_depth_consis_w

			t_d1, t_d2, t_d3 = deep_estimate_net(target_img)
			o_d1, o_d2, o_d3 = deep_estimate_net(source_img)

			loss_dehaza_consis = criterion_dehighlight(dehighlight_output_img, target_img)
			# loss_dehighlight_consis_w = criterion_dehighlight(dehighlight_output_img_2_depth_img, real_img_2_depth_img)
			loss_dehighlight_consis_w = criterion_dehighlight(dehighlight_output_img_2_depth_img, mask_img)
			loss_dehighlight_cr = criterion_dehighlight_cr(dehighlight_output_img, target_img, source_img)
			loss_dehighlight_u1 = criterion_dehighlight(t_d1, o_d1)
			loss_dehighlight_u2 = criterion_dehighlight(t_d2, o_d2)
			loss_dehighlight_u3 = criterion_dehighlight(t_d3, o_d3)

			loss_dehighlight_total = loss_dehaza_consis + 0.1*loss_dehighlight_consis_w + loss_dehighlight_cr + \
								loss_dehighlight_u1 + loss_dehighlight_u2 + loss_dehighlight_u3

		dehighlight_loss.update(loss_dehighlight_total.item())
		Detection_loss.update(loss_total_depth.item())

		optimizer_dehighlight.zero_grad()
		optimizer_depth.zero_grad()

		scaler.scale(loss_dehighlight_total + loss_total_depth).backward()

		scaler.step(optimizer_dehighlight)
		scaler.step(optimizer_depth)

		scaler.update()

	return dehighlight_loss.avg, Detection_loss.avg

def valid(val_loader, network):

	PSNR = AverageMeter()
	SSIM = AverageMeter()

	torch.cuda.empty_cache()
	network.eval()

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():
			output = network(source_img)[0].clamp_(-1, 1)
		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), source_img.size(0))
		# ssim_value = ssim(output, target_img, data_range=1.0, size_average=True)
		ssim_value = ssim(output, target_img, size_average=True)
		SSIM.update(ssim_value.item(), source_img.size(0))

	return PSNR.avg, SSIM.avg

if __name__ == '__main__':

	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', args.exp, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	# with torch.no_grad():
	# 	model_path = os.path.join("./depth/models", 'RA-Depth')
	# 	assert os.path.isdir(model_path), \
	# 		"Cannot find a folder at {}".format(model_path)
	# 	print("-> Loading weights from {}".format(model_path))
	#
	# 	encoder_path = os.path.join(model_path, "encoder.pth")
	# 	decoder_path = os.path.join(model_path, "depth.pth")
	# 	encoder_dict = torch.load(encoder_path)
	# 	encoder = hrnet18(False)
	# 	depth_decoder = DepthDecoder_MSF(encoder.num_ch_enc, [0], num_output_channels=1)
	# 	model_dict = encoder.state_dict()
	# 	encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
	# 	depth_decoder.load_state_dict(torch.load(decoder_path))
	# 	encoder.cuda()
	# 	encoder.eval()
	# 	depth_decoder.cuda()
	# 	depth_decoder.eval()

	network = DIACMPN_dehighlight_Indoor().cuda()
	DEPTH_NET = DepthNet.DN().cuda()
	deep_estimate_net = UNet().cuda()
	criterion_dehighlight = nn.L1Loss()
	criterion_dehighlight_cr = crloss()
	# criterion_depth = nn.L1Loss()
	criterion_detection = DetectionLoss()

	if setting['optimizer'] == 'adam':
		optimizer_dehighlight = torch.optim.Adam(network.parameters(), lr=setting['lr_dehighlight'])
		optimizer_detection = torch.optim.Adam(DEPTH_NET.parameters(), lr=setting['lr_detection'])
	elif setting['optimizer'] == 'adamw':
		optimizer_dehighlight = torch.optim.AdamW(network.parameters(), lr=setting['lr_dehighlight'])
		optimizer_detection = torch.optim.AdamW(DEPTH_NET.parameters(), lr=setting['lr_detection'])
	else:
		raise Exception("ERROR: unsupported optimizer")

	scheduler_dehighlight = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dehighlight, T_max=setting['epochs'], eta_min=setting['lr_dehighlight'] * 1e-2)
	# TODO
	scheduler_detection = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_detection, T_max=setting['epochs'], eta_min=setting['lr_detection'] * 1e-2)
	scaler = GradScaler()

	# # 分别计算参数量和计算量
	# macs1, params1 = profile(network, inputs=(input_tensor,))
	# macs2, params2 = profile(DEPTH_NET, inputs=(depth_input,))
	# macs3, params3 = profile(deep_estimate_net, inputs=(unet_input,))
	#
	# # 格式化输出
	# macs1, params1 = clever_format([macs1, params1], "%.3f")
	# macs2, params2 = clever_format([macs2, params2], "%.3f")
	# macs3, params3 = clever_format([macs3, params3], "%.3f")
	#
	# print(f"DIACMPN_dehighlight_Indoor 计算量: {macs1}, 参数量: {params1}")
	# print(f"DepthNet 计算量: {macs2}, 参数量: {params2}")
	# print(f"UNet 计算量: {macs3}, 参数量: {params3}")

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	train_dataset = PairLoaderForTrain(dataset_dir, 'train', 'train',
								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])

	# # 初始化基准形状为 None
	# reference_shapes = None
	#
	# # 定义预期的形状
	# expected_shapes = {
	# 	"source": (3, 256, 256),
	# 	"mask": (3, 512, 512),
	# 	"target": (3, 256, 256)
	# }

	# for idx, data in enumerate(train_dataset):
	# 	if isinstance(data, dict):  # 确保数据是字典
	# 		# 提取当前样本的形状信息
	# 		current_shapes = {key: value.shape for key, value in data.items() if hasattr(value, "shape")}
	#
	# 		# 检查是否与预期形状一致
	# 		inconsistent_keys = [
	# 			key for key in expected_shapes
	# 			if key in current_shapes and current_shapes[key] != expected_shapes[key]
	# 		]
	#
	# 		if inconsistent_keys:  # 如果存在不一致的键，输出详细信息
	# 			print(f"Sample {idx}: Shape inconsistency detected!")
	# 			for key in inconsistent_keys:
	# 				print(f"  Key: {key}, Expected Shape: {expected_shapes[key]}, Current Shape: {current_shapes[key]}")

	train_loader = DataLoader(train_dataset,
							  batch_size=setting['batch_size'],
							  shuffle=True,
							  num_workers=args.num_workers,
							  pin_memory=True,
							  drop_last=True)
	val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
							  setting['patch_size'])
	val_loader = DataLoader(val_dataset,
							batch_size=setting['batch_size'],
							num_workers=args.num_workers,
							pin_memory=True)

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)

	if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):

		print('==> Start training, current model name: ' + args.model)
		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

		best_psnr = 0

		for epoch in tqdm(range(setting['epochs'] + 1)):

			dehighlight_loss, detection_loss = train(train_loader, network, DEPTH_NET, criterion_dehighlight, criterion_dehighlight_cr, criterion_detection,
				   optimizer_dehighlight, optimizer_detection, scaler)

			writer.add_scalar('dehighlight_train_loss', dehighlight_loss, epoch)
			writer.add_scalar('detection_train_loss', detection_loss, epoch)

			print('')
			print('Dehighlight_loss: ', dehighlight_loss, epoch)
			print('Detection_loss: ', detection_loss, epoch)
			scheduler_dehighlight.step()
			scheduler_detection.step()

			if epoch % setting['eval_freq'] == 0:
				avg_psnr, avg_ssim = valid(val_loader, network)
				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				print('valid_psnr', avg_psnr, epoch)
				print('valid_ssim', avg_ssim, epoch)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					Dehaze_state_dict = {'dehighlight_net': network.state_dict(), 'dehighlight_optimizer': optimizer_dehighlight.state_dict(), 'epoch_dehighlight': epoch, 'dehighlight_net_best_psnr':best_psnr}
					depth_state_dict = {'detection_net': DEPTH_NET.state_dict(), 'detection_optimizer': optimizer_detection.state_dict(),'epoch_detection': epoch}
					torch.save(Dehaze_state_dict, os.path.join(save_dir, args.model+'.pth'))
					torch.save(depth_state_dict, os.path.join(save_dir, args.model_depth + '.pth'))
				print('best_psnr', best_psnr, epoch)

				writer.add_scalar('best_psnr', best_psnr, epoch)

				with open('./checkpoint/psnr_loss.txt', 'a') as file:
					file.write('Epoch [{}/{}], Loss_Dehighlight: {:.4f}, Loss_Detection: {:.4f}'
							   .format(epoch + 1, setting['epochs'], dehighlight_loss, dehighlight_loss))
					file.write('Best PSNR: {:.4f}\n'.format(best_psnr))
					file.write('Val PSNR: {:.4f}\n'.format(avg_psnr))
					file.write('Val SSIM: {:.4f}\n'.format(avg_ssim))
					file.write('\n')
	else:
		print('==> 1')

#############################################################################################################################

		dehighlight_checkpoint = torch.load(os.path.join(save_dir, args.model + '.pth'))
		depth_checkpoint = torch.load(os.path.join(save_dir, args.model_depth + '.pth'))
		network.load_state_dict(dehighlight_checkpoint['dehighlight_net'])
		DEPTH_NET.load_state_dict(depth_checkpoint['detection_net'])
		optimizer_dehighlight.load_state_dict(dehighlight_checkpoint['dehighlight_optimizer'])
		optimizer_detection.load_state_dict(depth_checkpoint['detection_optimizer'])
		best_psnr = dehighlight_checkpoint['dehighlight_net_best_psnr']
		start_epoch = dehighlight_checkpoint['epoch_dehighlight'] + 1

		print('Load start_epoch {} ！'.format(start_epoch))
		print('Load detection_optimizer {} ！'.format(optimizer_dehighlight))
		print('Load detection_optimizer {} ！'.format(optimizer_detection))

		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))
		# best_psnr = 0

		for epoch in tqdm(range(start_epoch, setting['epochs'] + 1)):
			dehighlight_loss, detection_loss = train(train_loader, network, DEPTH_NET, criterion_dehighlight, criterion_dehighlight_cr, criterion_detection,
				   optimizer_dehighlight, optimizer_detection, scaler)

			writer.add_scalar('dehighlight_train_loss', dehighlight_loss, epoch)
			writer.add_scalar('detection_train_loss', detection_loss, epoch)

			print('')
			print('Dehighlight_loss: ', dehighlight_loss, epoch)
			print('Detection_loss: ', detection_loss, epoch)
			scheduler_dehighlight.step()
			scheduler_detection.step()

			if epoch % setting['eval_freq'] == 0:
				avg_psnr, avg_ssim = valid(val_loader, network)

				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				print('valid_psnr', avg_psnr, epoch)
				print('valid_ssim', avg_ssim, epoch)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					Dehaze_state_dict = {'dehighlight_net': network.state_dict(),
										 'dehighlight_optimizer': optimizer_dehighlight.state_dict(),
										 'epoch_dehighlight': epoch,
										 'loss_dehighlight': dehighlight_loss,
										 'dehighlight_net_best_psnr':best_psnr}
					depth_state_dict = {'detection_net': DEPTH_NET.state_dict(),
										'detection_optimizer': optimizer_detection.state_dict(),
										'epoch_depth': epoch,
										'loss_detection': detection_loss}
					torch.save(Dehaze_state_dict, os.path.join(save_dir, args.model + '.pth'))
					torch.save(depth_state_dict, os.path.join(save_dir, args.model_depth + '.pth'))

				writer.add_scalar('best_psnr', best_psnr, epoch)
				print('best_psnr', best_psnr, epoch)

				with open('./checkpoint/psnr_loss.txt', 'a') as file:
					file.write('Epoch [{}/{}], Loss_Dehighlight: {:.4f}, Loss_Detection: {:.4f}'
							   .format(epoch + 1, setting['epochs'], dehighlight_loss, detection_loss))
					file.write('Best PSNR: {:.4f}\n'.format(best_psnr))
					file.write('Val PSNR: {:.4f}\n'.format(avg_psnr))
					file.write('Val SSIM: {:.4f}\n'.format(avg_ssim))
					file.write('\n')
