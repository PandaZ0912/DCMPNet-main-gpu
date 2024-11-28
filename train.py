import os
import argparse
import json
import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models import DepthNet, DIACMPN_dehighlight
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
parser.add_argument('--model_detection', default='Highlight-Removal-detection', type=str, help='detection model name')
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

def train(train_loader, network, DEPTH_NET, criterion_dehighlight, criterion_dehighlight_cr, criterion_detection,
				   optimizer_dehighlight, optimizer_detection, scaler):
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
			dehighlight_output_img_2_detection_img = DEPTH_NET(dehighlight_output_img)

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
			weighted_detection_output_img = torch.mul(dehighlight_output_img_2_detection_img, diff_dehighlight_w)
			weighted_real_img_2_detection_img = torch.mul(mask_img, diff_dehighlight_w)

			loss_detection_consis = criterion_detection(weighted_detection_output_img, weighted_real_img_2_detection_img)
			loss_detection_consis_w = criterion_detection(dehighlight_output_img_2_detection_img, mask_img)
			loss_total_detection = loss_detection_consis + loss_detection_consis_w

			t_d1, t_d2, t_d3 = deep_estimate_net(target_img)
			o_d1, o_d2, o_d3 = deep_estimate_net(source_img)

			loss_dehighlight_consis = criterion_dehighlight(dehighlight_output_img, target_img)
			loss_dehighlight_consis_w = criterion_dehighlight(dehighlight_output_img_2_detection_img, mask_img)
			loss_dehighlight_cr = criterion_dehighlight_cr(dehighlight_output_img, target_img, source_img)
			loss_dehighlight_u1 = criterion_dehighlight(t_d1, o_d1)
			loss_dehighlight_u2 = criterion_dehighlight(t_d2, o_d2)
			loss_dehighlight_u3 = criterion_dehighlight(t_d3, o_d3)

			loss_dehighlight_total = loss_dehighlight_consis + 0.1*loss_dehighlight_consis_w + loss_dehighlight_cr + \
								loss_dehighlight_u1 + loss_dehighlight_u2 + loss_dehighlight_u3

		dehighlight_loss.update(loss_dehighlight_total.item())
		Detection_loss.update(loss_total_detection.item())

		optimizer_dehighlight.zero_grad()
		optimizer_detection.zero_grad()

		scaler.scale(loss_dehighlight_total + loss_total_detection).backward()

		scaler.step(optimizer_dehighlight)
		scaler.step(optimizer_detection)

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
		ssim_value = ssim(output, target_img, size_average=True)
		SSIM.update(ssim_value.item(), source_img.size(0))

	return PSNR.avg, SSIM.avg

if __name__ == '__main__':

	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', args.exp, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	network = DIACMPN_dehighlight().cuda()
	DEPTH_NET = DepthNet.DN().cuda()
	deep_estimate_net = UNet().cuda()

	criterion_dehighlight = nn.L1Loss()
	criterion_dehighlight_cr = crloss()
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

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	train_dataset = PairLoaderForTrain(dataset_dir, 'train', 'train',
								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
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
					Dehighlight_state_dict = {'dehighlight_net': network.state_dict(), 'dehighlight_optimizer': optimizer_dehighlight.state_dict(), 'epoch_dehighlight': epoch, 'dehighlight_net_best_psnr':best_psnr}
					detection_state_dict = {'detection_net': DEPTH_NET.state_dict(), 'detection_optimizer': optimizer_detection.state_dict(),'epoch_detection': epoch}
					torch.save(Dehighlight_state_dict, os.path.join(save_dir, args.model+'.pth'))
					torch.save(detection_state_dict, os.path.join(save_dir, args.model_detection + '.pth'))
				print('best_psnr', best_psnr, epoch)

				writer.add_scalar('best_psnr', best_psnr, epoch)

				with open('./checkpoint/psnr_loss.txt', 'a') as file:
					file.write('Epoch [{}/{}], Loss_Dehighlight: {:.4f}, Loss_Detection: {:.4f}'
							   .format(epoch + 1, setting['epochs'], dehighlight_loss, detection_loss))
					file.write('Best PSNR: {:.4f}\n'.format(best_psnr))
					file.write('Val PSNR: {:.4f}\n'.format(avg_psnr))
					file.write('Val SSIM: {:.4f}\n'.format(avg_ssim))
					file.write('\n')

	else:
		print('==> 1')
		dehighlight_checkpoint = torch.load(os.path.join(save_dir, args.model + '.pth'))
		detection_checkpoint = torch.load(os.path.join(save_dir, args.model_detection + '.pth'))
		network.load_state_dict(dehighlight_checkpoint['dehighlight_net'])
		DEPTH_NET.load_state_dict(detection_checkpoint['detection_net'])
		optimizer_dehighlight.load_state_dict(dehighlight_checkpoint['dehighlight_optimizer'])
		optimizer_detection.load_state_dict(detection_checkpoint['detection_optimizer'])
		best_psnr = dehighlight_checkpoint['dehighlight_net_best_psnr']
		start_epoch = dehighlight_checkpoint['epoch_dehighlight'] + 1

		print('Load start_epoch {} ！'.format(start_epoch))
		print('Load detection_optimizer {} ！'.format(optimizer_dehighlight))
		print('Load detection_optimizer {} ！'.format(optimizer_detection))

		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

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
					Dehighlight_state_dict = {'dehighlight_net': network.state_dict(),
										 'dehighlight_optimizer': optimizer_dehighlight.state_dict(),
										 'epoch_dehighlight': epoch,
										 'loss_dehighlight': dehighlight_loss,
										 'dehighlight_net_best_psnr':best_psnr}
					detection_state_dict = {'detection_net': DEPTH_NET.state_dict(),
										'detection_optimizer': optimizer_detection.state_dict(),
										'epoch_detection': epoch,
										'loss_detection': detection_loss}
					torch.save(Dehighlight_state_dict, os.path.join(save_dir, args.model + '.pth'))
					torch.save(detection_state_dict, os.path.join(save_dir, args.model_detection + '.pth'))

				writer.add_scalar('best_psnr', best_psnr, epoch)
				print('best_psnr', best_psnr, epoch)

				with open('./checkpoint/psnr_loss.txt', 'a') as file:
					file.write('Epoch [{}/{}], Loss_Dehighlight: {:.4f}, Loss_Detection: {:.4f}'
							   .format(epoch + 1, setting['epochs'], dehighlight_loss, detection_loss))
					file.write('Best PSNR: {:.4f}\n'.format(best_psnr))
					file.write('Val PSNR: {:.4f}\n'.format(avg_psnr))
					file.write('Val SSIM: {:.4f}\n'.format(avg_ssim))
					file.write('\n')
