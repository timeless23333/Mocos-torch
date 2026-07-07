import argparse
import collections
import copy
import gc
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from utils import process_SG as process


def parse_args():
	parser = argparse.ArgumentParser(description="PyTorch implementation of MoCos without TensorFlow.")
	parser.add_argument("--dataset", default="KS20", choices=["IAS", "KGBD", "KS20", "BIWI", "CASIA_B"])
	parser.add_argument("--length", default="6")
	parser.add_argument("--lr", default=0.00035, type=float)
	parser.add_argument("--probe", default="probe")
	parser.add_argument("--gpu", default="0")
	parser.add_argument("--probe_type", default="", help="probe.gallery for CASIA_B, e.g. nm.nm")
	parser.add_argument("--patience", default=150, type=int)
	parser.add_argument("--mode", default="Train", choices=["Train", "Eval"])
	parser.add_argument("--save_flag", default="0")
	parser.add_argument("--save_model", default="0")
	parser.add_argument("--batch_size", default=256, type=int)
	parser.add_argument("--epochs", default=15000, type=int)
	parser.add_argument("--H", default=128, type=int)
	parser.add_argument("--n_heads", default=8, type=int)
	parser.add_argument("--L_transformer", default=2, type=int)
	parser.add_argument("--fusion_lambda", default=None, type=float)
	parser.add_argument("--t_1", default=0.1, type=float)
	parser.add_argument("--t_2", default=10.0, type=float)
	parser.add_argument("--pos_enc", default="1")
	parser.add_argument("--enc_k", default=10, type=int)
	parser.add_argument("--rand_flip", default=None)
	parser.add_argument("--prob_t", default=None, type=float)
	parser.add_argument("--prob_s", default=None, type=float)
	parser.add_argument("--device", default="", help="Override device, e.g. cpu or cuda:0")
	parser.add_argument("--seed", default=None, type=int, help="Set Python, NumPy, and PyTorch RNG seeds.")
	parser.add_argument("--deterministic", default="0", help="Use deterministic CuDNN/PyTorch behavior when possible.")
	parser.add_argument("--resume", default="", help="Resume training from a checkpoint path, or use 'auto' for last.pt.")
	return parser.parse_args()


def apply_dataset_defaults(args):
	if args.dataset == "CASIA_B":
		args.length = "40"
		if args.rand_flip is None:
			args.rand_flip = "0"
		if args.fusion_lambda is None:
			args.fusion_lambda = 1.0
		if args.patience == 150:
			args.patience = 100
	elif args.dataset == "KGBD":
		if args.rand_flip is None:
			args.rand_flip = "0"
		if args.prob_s is None:
			args.prob_s = 0.5
		if args.prob_t is None:
			args.prob_t = 0.25
		if args.fusion_lambda is None:
			args.fusion_lambda = 0.9
	elif args.dataset == "IAS":
		if args.rand_flip is None:
			args.rand_flip = "0"
		if args.probe == "A":
			if args.prob_s is None:
				args.prob_s = 0.5
			if args.prob_t is None:
				args.prob_t = 0.1
			if args.fusion_lambda is None:
				args.fusion_lambda = 0.75
		elif args.probe == "B":
			if args.prob_s is None:
				args.prob_s = 0.25
			if args.prob_t is None:
				args.prob_t = 0.25
			if args.fusion_lambda is None:
				args.fusion_lambda = 0.75
	elif args.dataset == "BIWI":
		if args.probe in ["Walking", "Still"]:
			if args.prob_s is None:
				args.prob_s = 0.25
			if args.prob_t is None:
				args.prob_t = 0.25
			if args.fusion_lambda is None:
				args.fusion_lambda = 0.9 if args.probe == "Walking" else 0.25
	elif args.dataset == "KS20":
		if args.prob_s is None:
			args.prob_s = 0.25
		if args.prob_t is None:
			args.prob_t = 0.25
		if args.fusion_lambda is None:
			args.fusion_lambda = 0.9
	if args.rand_flip is None:
		args.rand_flip = "1"
	if args.prob_s is None:
		args.prob_s = 0.5
	if args.prob_t is None:
		args.prob_t = 0.0
	if args.fusion_lambda is None:
		args.fusion_lambda = 0.5
	return args


def nb_nodes_for(dataset):
	if dataset == "KS20":
		return 25
	if dataset == "CASIA_B":
		return 14
	return 20


def onehot_to_index(labels):
	return np.argmax(np.asarray(labels), axis=-1).astype(np.int64)


def set_random_seed(seed, deterministic=False):
	if seed is None:
		return
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	if deterministic:
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		try:
			torch.use_deterministic_algorithms(True)
		except Exception as exc:
			print("warning: deterministic algorithms not fully enabled:", exc)


def k_hop_adj(adj, k):
	adj = np.asarray(adj, dtype=np.float32)
	reach = np.eye(adj.shape[0], dtype=np.float32)
	step = np.eye(adj.shape[0], dtype=np.float32)
	base = (adj > 0).astype(np.float32)
	for _ in range(k):
		step = (step @ (base + np.eye(adj.shape[0], dtype=np.float32)) > 0).astype(np.float32)
		reach = np.maximum(reach, step)
	return reach


def clique_adj(node_num, groups):
	adj = np.zeros((node_num, node_num), dtype=np.float32)
	for group in groups:
		for i in group:
			for j in group:
				if i != j:
					adj[i, j] = 1.0
	return adj


def build_motif_adjs(dataset, adj_joint):
	adj1 = np.asarray(adj_joint[0] if adj_joint.ndim == 3 else adj_joint, dtype=np.float32)
	node_num = adj1.shape[0]
	adj2 = k_hop_adj(adj1, 2)
	adj3 = k_hop_adj(adj1, 3)

	if node_num == 20:
		adj4 = np.zeros((20, 20), dtype=np.float32)
		adj4[8, [9, 10, 11]] = 1
		adj4[9, [8, 10, 11]] = 1
		adj4[10, [9, 8, 11]] = 1
		adj4[11, [9, 10, 8]] = 1
		for i in [8, 9, 10, 11]:
			for j in [4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15]:
				adj4[i, j] = 1
		adj4[4, [5, 6, 7]] = 1
		adj4[5, [4, 6, 7]] = 1
		adj4[6, [5, 4, 7]] = 1
		adj4[7, [5, 6, 4]] = 1
		for i in [4, 5, 6, 7]:
			for j in [8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15]:
				adj4[i, j] = 1

		adj5 = np.zeros((20, 20), dtype=np.float32)
		adj5[16, [17, 18, 19]] = 1
		adj5[17, [16, 18, 19]] = 1
		adj5[18, [16, 17, 19]] = 1
		adj5[19, [16, 18, 17]] = 1
		for i in [16, 17, 18, 19]:
			for j in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
				adj5[i, j] = 1
		adj5[12, [13, 14, 15]] = 1
		adj5[13, [12, 14, 15]] = 1
		adj5[14, [12, 13, 15]] = 1
		adj5[15, [12, 13, 14]] = 1
		for i in [12, 13, 14, 15]:
			for j in [4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19]:
				adj5[i, j] = 1
	else:
		if node_num == 25:
			arms = [[4, 5, 6, 7, 21, 22], [8, 9, 10, 11, 23, 24]]
			legs = [[12, 13, 14, 15], [16, 17, 18, 19]]
		elif node_num == 14:
			arms = [[2, 3, 4], [5, 6, 7]]
			legs = [[8, 9, 10], [11, 12, 13]]
		else:
			arms = [[4, 5, 6, 7], [8, 9, 10, 11]]
			legs = [[12, 13, 14, 15], [16, 17, 18, 19]]
		adj4 = clique_adj(node_num, arms)
		adj5 = clique_adj(node_num, legs)
	for adj in [adj1, adj2, adj3, adj4, adj5]:
		np.fill_diagonal(adj, 0.0)
	return np.stack([adj1, adj2, adj3, adj4, adj5], axis=0).astype(np.float32)


class MGTLayer(nn.Module):
	def __init__(self, hidden_size, num_heads, motif_adjs):
		super().__init__()
		if hidden_size % num_heads != 0:
			raise ValueError("H must be divisible by n_heads.")
		self.hidden_size = hidden_size
		self.num_heads = num_heads
		self.head_dim = hidden_size // num_heads
		self.q_layers = nn.ModuleList([nn.Linear(hidden_size, self.head_dim, bias=False) for _ in range(num_heads)])
		self.k_layers = nn.ModuleList([nn.Linear(hidden_size, self.head_dim, bias=False) for _ in range(num_heads)])
		self.v_layers = nn.ModuleList([nn.Linear(hidden_size, self.head_dim, bias=False) for _ in range(num_heads)])
		self.register_buffer("motif_adjs", torch.from_numpy(motif_adjs), persistent=False)

	def forward(self, x):
		heads = []
		scale = self.head_dim ** 0.5
		for i in range(self.num_heads):
			q = self.q_layers[i](x)
			k = self.k_layers[i](x)
			v = self.v_layers[i](x)
			scores = torch.matmul(q, k.transpose(-1, -2)) / scale
			if 1 <= i <= 5:
				scores = scores * self.motif_adjs[i - 1].to(scores.device)
			attn = F.softmax(torch.clamp(scores, min=-5.0, max=5.0), dim=-1)
			heads.append(torch.matmul(attn, v))
		return torch.cat(heads, dim=-1)


class MoCosTorch(nn.Module):
	def __init__(self, joint_num, pos_dim, hidden_size, num_heads, num_layers, motif_adjs, use_pos_enc=True):
		super().__init__()
		self.use_pos_enc = use_pos_enc
		self.input_proj = nn.Sequential(
			nn.Linear(3, hidden_size),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_size, hidden_size),
		)
		self.pos_proj = nn.Linear(pos_dim, hidden_size)
		self.layers = nn.ModuleList([MGTLayer(hidden_size, num_heads, motif_adjs) for _ in range(num_layers)])
		self.dropout1 = nn.Dropout(0.5)
		self.out_proj = nn.Linear(hidden_size, hidden_size)
		self.norm1 = nn.BatchNorm1d(hidden_size)
		self.ffn = nn.Sequential(
			nn.Linear(hidden_size, hidden_size * 2),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(hidden_size * 2, hidden_size),
		)
		self.norm2 = nn.BatchNorm1d(hidden_size)
		self.ssk_proj_all = nn.Linear(hidden_size, hidden_size, bias=False)
		self.ssk_proj_proto = nn.Linear(hidden_size, hidden_size, bias=False)

	def _bn4d(self, norm, x):
		shape = x.shape
		return norm(x.reshape(-1, shape[-1])).reshape(shape)

	def forward(self, x, pos_enc):
		h = self.input_proj(x)
		if self.use_pos_enc:
			pos = self.pos_proj(pos_enc).view(1, 1, pos_enc.shape[0], -1)
			h = h + pos
		for layer in self.layers:
			h = layer(h)
		h = h + self.out_proj(self.dropout1(h))
		h = self._bn4d(self.norm1, h)
		h = h + self.ffn(h)
		h = self._bn4d(self.norm2, h)
		spatial_h = h
		frame_features = spatial_h.mean(dim=2)
		seq_features = frame_features.mean(dim=1)
		return seq_features, frame_features, spatial_h

	def csp_loss(self, frame_features, spatial_features, labels, prototypes, seq_mask, node_mask, t1, t2, fusion_lambda):
		masked_nodes = spatial_features[:, :, node_mask, :]
		g_h = masked_nodes.mean(dim=2)
		g_h_seq = g_h[:, seq_mask, :].mean(dim=1)

		str_logits = torch.matmul(F.normalize(g_h_seq, dim=-1), F.normalize(prototypes, dim=-1).t()) / t1
		str_loss = F.cross_entropy(str_logits, labels)

		all_ftr = self.ssk_proj_all(g_h)
		proto_ftr = self.ssk_proj_proto(prototypes)
		ssk_logits = torch.matmul(all_ftr, proto_ftr.t()) / t2
		ssk_labels = labels.view(-1, 1).expand(-1, g_h.shape[1]).reshape(-1)
		ssk_loss = F.cross_entropy(ssk_logits.reshape(-1, ssk_logits.shape[-1]), ssk_labels, reduction="none")
		ssk_loss = ssk_loss.view(g_h.shape[0], g_h.shape[1]).sum(dim=1).mean()

		loss = (1.0 - fusion_lambda) * ssk_loss + fusion_lambda * str_loss
		return loss, str_loss.detach(), ssk_loss.detach()


@torch.no_grad()
def extract_features(model, data, labels, pos_enc, batch_size, device):
	model.eval()
	features = []
	label_rows = []
	for start in range(0, len(data) - batch_size + 1, batch_size):
		x = torch.from_numpy(data[start:start + batch_size]).float().to(device)
		seq_ftr, _, _ = model(x, pos_enc)
		features.append(seq_ftr.cpu())
		label_rows.extend(labels[start:start + batch_size].tolist())
	if not features:
		return torch.empty(0, device="cpu"), np.empty((0,))
	return torch.cat(features, dim=0), np.asarray(label_rows)


def generate_class_prototypes(labels, features, nb_classes, device):
	centers = collections.defaultdict(list)
	for i, label in enumerate(labels):
		centers[int(label)].append(features[i])
	prototypes = []
	for label in range(nb_classes):
		if centers[label]:
			prototypes.append(torch.stack(centers[label], dim=0).mean(dim=0))
		else:
			prototypes.append(torch.zeros(features.shape[1], dtype=features.dtype))
	return torch.stack(prototypes, dim=0).to(device)


def random_mask(size, drop_prob, device):
	mask = torch.rand(size, device=device) >= drop_prob
	while not bool(mask.any()):
		mask = torch.rand(size, device=device) >= drop_prob
	return mask


def mean_ap(distmat, query_ids, gallery_ids, skip_self=False):
	indices = np.argsort(distmat, axis=1)
	matches = gallery_ids[indices] == query_ids[:, np.newaxis]
	aps = []
	start = 1 if skip_self else 0
	for i in range(start, distmat.shape[0]):
		valid = np.ones(indices.shape[1], dtype=bool)
		if skip_self:
			valid[0] = False
		y_true = matches[i, valid]
		y_score = -distmat[i][indices[i]][valid]
		y_score[np.isnan(y_score)] = 0
		if np.any(y_true):
			aps.append(average_precision_score(y_true, y_score))
	if not aps:
		return 0.0
	return float(np.mean(aps))


def evaluate_features(gallery_features, gallery_labels, probe_features, probe_labels, probe_type=""):
	x = gallery_features.float()
	t_x = probe_features.float()
	dist = torch.cdist(t_x, x, p=2).cpu().numpy()
	sort_idx = np.argsort(dist, axis=1)
	skip_self = probe_type in ["nm.nm", "cl.cl", "bg.bg"]
	m_ap = mean_ap(dist, probe_labels, gallery_labels, skip_self=skip_self)
	top_1 = top_5 = top_10 = 0
	for i in range(sort_idx.shape[0]):
		offset = 1 if skip_self else 0
		if probe_labels[i] in gallery_labels[sort_idx[i, offset:offset + 1]]:
			top_1 += 1
		if probe_labels[i] in gallery_labels[sort_idx[i, offset:offset + 5]]:
			top_5 += 1
		if probe_labels[i] in gallery_labels[sort_idx[i, offset:offset + 10]]:
			top_10 += 1
	total = max(sort_idx.shape[0], 1)
	return m_ap, top_1 / total, top_5 / total, top_10 / total


def load_split(args, split, batch_size, nb_nodes):
	return process.gen_train_data(
		dataset=args.dataset,
		split=split,
		time_step=int(args.length),
		nb_nodes=nb_nodes,
		nhood=1,
		global_att=False,
		batch_size=batch_size,
		enc_k=args.enc_k,
	)


def load_initial_data(args, nb_nodes):
	if args.probe_type:
		from utils import process_cme_SG as cme_process
		return cme_process.gen_train_data(
			dataset=args.dataset,
			split=args.probe,
			time_step=int(args.length),
			nb_nodes=nb_nodes,
			nhood=1,
			global_att=False,
			batch_size=args.batch_size,
			PG_type=args.probe_type.split(".")[0],
		)
	return load_split(args, args.probe, args.batch_size, nb_nodes)


def load_gallery_data(args, nb_nodes):
	if args.dataset in ["KGBD", "KS20"]:
		return load_split(args, "gallery", args.batch_size, nb_nodes)
	if args.dataset == "BIWI":
		return load_split(args, "Still" if args.probe == "Walking" else "Walking", args.batch_size, nb_nodes)
	if args.dataset == "IAS":
		return load_split(args, "B" if args.probe == "A" else "A", args.batch_size, nb_nodes)
	if args.dataset == "CASIA_B":
		from utils import process_cme_SG as cme_process
		return cme_process.gen_train_data(
			dataset=args.dataset,
			split=args.probe,
			time_step=int(args.length),
			nb_nodes=nb_nodes,
			nhood=1,
			global_att=False,
			batch_size=args.batch_size,
			PG_type=args.probe_type.split(".")[1],
		)
	raise ValueError(args.dataset)


def checkpoint_path(args):
	change = "_CME" if args.probe_type else ""
	change += "_MoCos_Torch_f_%s_prob_s_%s_prob_t_%s_lambda_%s" % (
		args.length,
		args.prob_s,
		args.prob_t,
		args.fusion_lambda,
	)
	name = "%s_best.pt" % args.probe_type if args.dataset == "CASIA_B" and args.probe_type else "best.pt"
	return os.path.join("ReID_Models", args.dataset, args.probe, change, name)


def last_checkpoint_path(args):
	best_path = checkpoint_path(args)
	name = "%s_last.pt" % args.probe_type if args.dataset == "CASIA_B" and args.probe_type else "last.pt"
	return os.path.join(os.path.dirname(best_path), name)


def load_torch_checkpoint(path, device):
	try:
		return torch.load(path, map_location=device, weights_only=False)
	except TypeError:
		return torch.load(path, map_location=device)


def checkpoint_state(model, optimizer, args, epoch, best_map, best_top_1, cur_patience):
	return {
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"args": vars(args),
		"epoch": epoch,
		"best_map": best_map,
		"best_top_1": best_top_1,
		"cur_patience": cur_patience,
	}


def load_resume_checkpoint(args, model, optimizer, device):
	if not args.resume:
		return 0, 0.0, -1.0, 0
	resume_path = last_checkpoint_path(args) if args.resume == "auto" else args.resume
	state = load_torch_checkpoint(resume_path, device)
	model_state = state["model"] if isinstance(state, dict) and "model" in state else state
	missing, unexpected = model.load_state_dict(model_state, strict=False)
	if missing or unexpected:
		print("resume loaded with missing keys: %d | unexpected keys: %d" % (len(missing), len(unexpected)))
	if isinstance(state, dict) and "optimizer" in state:
		try:
			optimizer.load_state_dict(state["optimizer"])
		except ValueError as exc:
			print("warning: optimizer state not restored:", exc)
	start_epoch = int(state.get("epoch", -1)) + 1 if isinstance(state, dict) else 0
	best_map = float(state.get("best_map", 0.0)) if isinstance(state, dict) else 0.0
	best_top_1 = float(state.get("best_top_1", -1.0)) if isinstance(state, dict) else -1.0
	cur_patience = int(state.get("cur_patience", 0)) if isinstance(state, dict) else 0
	print("resumed:", resume_path, "| start epoch:", start_epoch)
	return start_epoch, best_map, best_top_1, cur_patience


def main():
	args = apply_dataset_defaults(parse_args())
	if args.dataset != "CASIA_B" and args.length not in ["4", "6", "8", "10"]:
		raise ValueError("length must be one of 4, 6, 8, 10 for non-CASIA_B datasets.")
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
	set_random_seed(args.seed, args.deterministic == "1")
	nb_nodes = nb_nodes_for(args.dataset)

	data_tuple = load_initial_data(args, nb_nodes)
	X_train_J, _, _, _, _, y_train, X_test_J, _, _, _, _, y_test, adj_J, _, pos_enc_ori, *rest = data_tuple
	nb_classes = rest[-1]
	joint_num = X_train_J.shape[2]
	motif_adjs = build_motif_adjs(args.dataset, adj_J)
	pos_enc = torch.from_numpy(pos_enc_ori).float().to(device)

	model = MoCosTorch(
		joint_num=joint_num,
		pos_dim=pos_enc.shape[1],
		hidden_size=args.H,
		num_heads=args.n_heads,
		num_layers=args.L_transformer,
		motif_adjs=motif_adjs,
		use_pos_enc=args.pos_enc == "1",
	).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	path = checkpoint_path(args)
	last_path = last_checkpoint_path(args)

	print("----- PyTorch MoCos hyperparams -----")
	print("dataset: %s | probe: %s | length: %s | classes: %d" % (args.dataset, args.probe, args.length, nb_classes))
	print("H: %d | heads: %d | layers: %d | device: %s" % (args.H, args.n_heads, args.L_transformer, device))
	print("p_s: %.3f | p_t: %.3f | lambda: %.3f" % (args.prob_s, args.prob_t, args.fusion_lambda))
	if args.seed is not None:
		print("seed: %d | deterministic: %s" % (args.seed, args.deterministic))

	if args.mode == "Eval":
		state = load_torch_checkpoint(path, device)
		model.load_state_dict(state["model"], strict=False)
		gallery_tuple = load_gallery_data(args, nb_nodes)
		_, _, _, _, _, _, X_gal_J, _, _, _, _, y_gal, *_ = gallery_tuple
		gal_f, gal_l = extract_features(model, X_gal_J, onehot_to_index(y_gal), pos_enc, args.batch_size, device)
		pro_f, pro_l = extract_features(model, X_test_J, onehot_to_index(y_test), pos_enc, args.batch_size, device)
		m_ap, top_1, top_5, top_10 = evaluate_features(gal_f, gal_l, pro_f, pro_l, args.probe_type)
		print("[Evaluation] mAP: %.4f | R1: %.4f | R5: %.4f | R10: %.4f" % (m_ap, top_1, top_5, top_10))
		return

	gallery_tuple = load_gallery_data(args, nb_nodes)
	_, _, _, _, _, _, X_gal_J, _, _, _, _, y_gal, *_ = gallery_tuple
	del data_tuple, gallery_tuple
	gc.collect()

	train_labels_idx = onehot_to_index(y_train)
	gallery_labels_idx = onehot_to_index(y_gal)
	probe_labels_idx = onehot_to_index(y_test)
	start_epoch, best_map, best_top_1, cur_patience = load_resume_checkpoint(args, model, optimizer, device)

	for epoch in range(start_epoch, args.epochs):
		train_features, train_labels_seen = extract_features(model, X_train_J, train_labels_idx, pos_enc, args.batch_size, device)
		prototypes = generate_class_prototypes(train_labels_seen, train_features, nb_classes, device)

		model.train()
		losses = []
		for start in range(0, len(X_train_J) - args.batch_size + 1, args.batch_size):
			x = torch.from_numpy(X_train_J[start:start + args.batch_size]).float().to(device)
			labels = torch.from_numpy(train_labels_idx[start:start + args.batch_size]).long().to(device)
			seq_mask = random_mask(int(args.length), args.prob_t, device)
			node_mask = random_mask(joint_num, args.prob_s, device)
			current_pos = pos_enc
			if args.rand_flip == "1":
				sign = torch.where(torch.rand(pos_enc.shape[1], device=device) >= 0.5, 1.0, -1.0)
				current_pos = pos_enc * sign

			optimizer.zero_grad()
			_, frame_ftr, spatial_ftr = model(x, current_pos)
			loss, str_loss, ssk_loss = model.csp_loss(
				frame_ftr,
				spatial_ftr,
				labels,
				prototypes,
				seq_mask,
				node_mask,
				args.t_1,
				args.t_2,
				args.fusion_lambda,
			)
			loss.backward()
			optimizer.step()
			losses.append(float(loss.detach().cpu()))
			if (start // args.batch_size) % 20 == 0:
				print(
					"[%d] Batch %d | CSP %.5f | SSk %.5f | STr %.5f"
					% (epoch, start // args.batch_size, float(loss), float(ssk_loss), float(str_loss))
				)
		gal_f, gal_l = extract_features(model, X_gal_J, gallery_labels_idx, pos_enc, args.batch_size, device)
		pro_f, pro_l = extract_features(model, X_test_J, probe_labels_idx, pos_enc, args.batch_size, device)
		m_ap, top_1, top_5, top_10 = evaluate_features(gal_f, gal_l, pro_f, pro_l, args.probe_type)

		if top_1 > best_top_1:
			best_top_1 = top_1
			best_map = m_ap
			cur_patience = 0
			if args.save_model == "1":
				os.makedirs(os.path.dirname(path), exist_ok=True)
				torch.save(checkpoint_state(model, optimizer, args, epoch, best_map, best_top_1, cur_patience), path)
				print("saved:", path)
		else:
			cur_patience += 1

		if args.save_model == "1":
			os.makedirs(os.path.dirname(last_path), exist_ok=True)
			torch.save(checkpoint_state(model, optimizer, args, epoch, best_map, best_top_1, cur_patience), last_path)

		print(
			"[Epoch %d] loss: %.5f | mAP: %.4f | R1: %.4f | R5: %.4f | R10: %.4f | best mAP/R1: %.4f/%.4f | patience: %d/%d"
			% (epoch, np.mean(losses), m_ap, top_1, top_5, top_10, best_map, best_top_1, cur_patience, args.patience)
		)
		if cur_patience >= args.patience:
			break


if __name__ == "__main__":
	main()
