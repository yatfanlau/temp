import datetime as dt
import json
import os
import pickle
import re
import time
import warnings

import torch

try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	SummaryWriter = None


def _natural_key(value):
	return [int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", str(value))]


class Logger:

	def __init__(self, name, datetime=None, use_csv=True, use_tensorboard=False, run_path=None):
		"""
		Logger stores run configuration, scalar metrics and model checkpoints.
		:name: logging name (e.g. model name / dataset name / ...)
		:datetime: date and time of logging start (useful in case of multiple runs)
		:use_csv: log scalar values to plain text files
		:use_tensorboard: log scalar values to tensorboard
		:run_path: optional exact run directory
		"""
		self.name = name
		self.use_csv = use_csv
		self.use_tensorboard = use_tensorboard

		if run_path is None:
			self.datetime = datetime or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			self.run_path = os.path.join("Logger", self.name, self.datetime)
		else:
			self.run_path = run_path
			self.datetime = os.path.basename(os.path.normpath(run_path))
			parent = os.path.basename(os.path.dirname(os.path.normpath(run_path)))
			if parent:
				self.name = parent

		self.logs_path = os.path.join(self.run_path, "logs")
		self.plots_path = os.path.join(self.run_path, "plots")
		self.states_path = os.path.join(self.run_path, "states")

		if self.use_csv:
			os.makedirs(self.logs_path, exist_ok=True)
			os.makedirs(self.plots_path, exist_ok=True)

		if self.use_tensorboard:
			if SummaryWriter is None:
				warnings.warn("tensorboard is not installed; continuing without tensorboard event files")
				self.use_tensorboard = False
			else:
				directory = os.path.join(self.run_path, "tensorboard")
				os.makedirs(directory, exist_ok=True)
				self.writer = SummaryWriter(directory)

	def save_config(self, config):
		os.makedirs(self.run_path, exist_ok=True)
		path = os.path.join(self.run_path, "config.json")
		with open(path, "w", encoding="utf-8") as handle:
			json.dump(config, handle, indent=2, sort_keys=True)

	def load_config(self, run_path=None):
		path = os.path.join(run_path or self.run_path, "config.json")
		with open(path, "r", encoding="utf-8") as handle:
			return json.load(handle)

	def log(self, item, value, index):
		"""
		log index value couple for specific item into csv file / tensorboard
		:item: string describing item (e.g. "training_loss","test_loss")
		:value: value to log
		:index: index (e.g. batchindex / epoch)
		"""
		if self.use_csv:
			filename = os.path.join(self.logs_path, f"{item}.log")
			append_write = "a" if os.path.exists(filename) else "w"
			with open(filename, append_write, encoding="utf-8") as log_file:
				log_file.write(f"{index}, {value}\n")

		if self.use_tensorboard:
			self.writer.add_scalar(item, value, index)

	def log_histogram(self, item, values, index):
		"""
		log index values-histogram couple for specific item to tensorboard
		"""
		if self.use_tensorboard:
			self.writer.add_histogram(item, values, index)

	def log_model_gradients(self, item, model, index):
		"""
		log index model-gradients-histogram couple for specific item to tensorboard
		"""
		if self.use_tensorboard:
			params = [p for p in model.parameters()]
			if params:
				gradients = torch.cat([p.grad.view(-1) for p in params if p.grad is not None])
				self.writer.add_histogram(f"{item}_grad_histogram", gradients, index)
				self.writer.add_scalar(f"{item}_grad_norm2", gradients.norm(2), index)

	def plot(self, item, log=False, smoothing=0.025, ylim=None):
		"""
		plot item metrics
		:item: item
		:log: logarithmic scale. Default: False
		:smoothing: fraction used for moving-average smoothing. Default: 0.025
		:ylim: y-axis limits [lower,upper]
		"""
		if not self.use_csv:
			raise RuntimeError("set use_csv=True if you want to plot metrics")

		import matplotlib.pyplot as plt
		import numpy as np

		log_path = os.path.join(self.logs_path, f"{item}.log")
		train_loss = np.loadtxt(open(log_path, "rb"), delimiter=",")
		if train_loss.ndim == 1:
			train_loss = train_loss.reshape(1, -1)

		plt.figure(1, figsize=(12, 6))
		plt.clf()
		plt.title(self.name)
		plt.ylabel(item)
		plt.xlabel("index")

		x_values = train_loss[:, 0]
		y_values = train_loss[:, 1]

		if log:
			plt.semilogy(x_values, y_values, "r")
		else:
			plt.plot(x_values, y_values, "r")

		window = max(1, int(len(y_values) * smoothing))
		if window > 1:
			kernel = np.ones(window) / window
			smoothed = np.convolve(y_values, kernel, mode="same")
		else:
			smoothed = y_values

		if log:
			plt.semilogy(x_values, smoothed, "b")
		else:
			plt.plot(x_values, smoothed, "b")

		if not log:
			if ylim is not None:
				plt.ylim(ylim)
			elif len(smoothed) > 1:
				mean = np.mean(smoothed)
				std = np.std(smoothed)
				plt.ylim([mean - 2 * std, mean + 4 * std])

		filename = f"{item}_log.png" if log else f"{item}.png"
		plt.savefig(os.path.join(self.plots_path, filename), dpi=400)

	def save_state(self, model, optimizer, index="final"):
		"""
		saves state of model and optimizer
		:model: model to save (if list: save multiple models)
		:optimizer: optimizer (if list: save multiple optimizers)
		:index: index of state to save (e.g. specific epoch)
		"""
		os.makedirs(self.states_path, exist_ok=True)
		path = os.path.join(self.states_path, f"{index}.state")
		state = {}

		if not isinstance(model, list):
			model = [model]
		for i, module in enumerate(model):
			state[f"model{i}"] = module.state_dict()

		if optimizer is not None:
			if not isinstance(optimizer, list):
				optimizer = [optimizer]
			for i, opt in enumerate(optimizer):
				state[f"optimizer{i}"] = opt.state_dict()

		torch.save(state, path)

	def save_dict(self, dic, index="final"):
		"""
		saves dictionary - helpful to save arbitrary run state
		"""
		os.makedirs(self.states_path, exist_ok=True)
		path = os.path.join(self.states_path, f"{index}.dic")
		with open(path, "wb") as handle:
			pickle.dump(dic, handle)

	def _resolve_run_path(self, datetime=None, run_path=None):
		if run_path is not None:
			return run_path
		if datetime is not None:
			return os.path.join("Logger", self.name, datetime)

		base_path = os.path.join("Logger", self.name)
		if not os.path.isdir(base_path):
			raise FileNotFoundError(f"run base path not found: {base_path}")

		run_dirs = [
			entry
			for entry in os.listdir(base_path)
			if os.path.isdir(os.path.join(base_path, entry))
		]
		if not run_dirs:
			raise FileNotFoundError(f"no runs available in {base_path}")

		run_dirs = sorted(run_dirs, key=_natural_key)
		latest = run_dirs[-1]
		if latest == self.datetime and len(run_dirs) > 1:
			latest = run_dirs[-2]
		return os.path.join(base_path, latest)

	def _resolve_state_index(self, states_path, index):
		if index is not None:
			return str(index)

		state_files = [
			os.path.splitext(filename)[0]
			for filename in os.listdir(states_path)
			if filename.endswith(".state")
		]
		if not state_files:
			raise FileNotFoundError(f"no checkpoint states available in {states_path}")

		numeric = [name for name in state_files if name.isdigit()]
		if numeric:
			return sorted(numeric, key=lambda name: int(name))[-1]
		return sorted(state_files, key=_natural_key)[-1]

	def load_state(self, model, optimizer, datetime=None, index=None, continue_datetime=False, run_path=None):
		"""
		loads state of model and optimizer
		:model: model to load (if list: load multiple models)
		:optimizer: optimizer to load (if list: load multiple optimizers; if None: don't load)
		:datetime: date and time from run to load (if None: take latest folder)
		:index: index of state to load (if None: take latest index)
		:continue_datetime: if True, continue writing into the loaded run directory
		:run_path: exact run directory to load
		:return: datetime, index
		"""
		resolved_run_path = self._resolve_run_path(datetime=datetime, run_path=run_path)
		resolved_datetime = os.path.basename(os.path.normpath(resolved_run_path))
		states_path = os.path.join(resolved_run_path, "states")
		resolved_index = self._resolve_state_index(states_path, index)
		path = os.path.join(states_path, f"{resolved_index}.state")
		state = torch.load(path, map_location="cpu")

		if not isinstance(model, list):
			model = [model]
		for i, module in enumerate(model):
			module.load_state_dict(state[f"model{i}"])

		if optimizer is not None:
			if not isinstance(optimizer, list):
				optimizer = [optimizer]
			for i, opt in enumerate(optimizer):
				optimizer_key = f"optimizer{i}"
				if optimizer_key in state:
					opt.load_state_dict(state[optimizer_key])

		if continue_datetime:
			self.run_path = resolved_run_path
			self.datetime = resolved_datetime
			self.logs_path = os.path.join(self.run_path, "logs")
			self.plots_path = os.path.join(self.run_path, "plots")
			self.states_path = os.path.join(self.run_path, "states")

		return resolved_datetime, resolved_index

	def load_dict(self, dic, datetime=None, index=None, continue_datetime=False, run_path=None):
		"""
		loads dictionary state
		:return: datetime, index
		"""
		resolved_run_path = self._resolve_run_path(datetime=datetime, run_path=run_path)
		resolved_datetime = os.path.basename(os.path.normpath(resolved_run_path))
		states_path = os.path.join(resolved_run_path, "states")

		if index is None:
			dict_files = [
				os.path.splitext(filename)[0]
				for filename in os.listdir(states_path)
				if filename.endswith(".dic")
			]
			if not dict_files:
				raise FileNotFoundError(f"no dictionary states available in {states_path}")
			numeric = [name for name in dict_files if name.isdigit()]
			if numeric:
				index = sorted(numeric, key=lambda name: int(name))[-1]
			else:
				index = sorted(dict_files, key=_natural_key)[-1]

		path = os.path.join(states_path, f"{index}.dic")
		with open(path, "rb") as handle:
			state = pickle.load(handle)

		for key, value in state.items():
			dic[key] = value

		if continue_datetime:
			self.run_path = resolved_run_path
			self.datetime = resolved_datetime
			self.logs_path = os.path.join(self.run_path, "logs")
			self.plots_path = os.path.join(self.run_path, "plots")
			self.states_path = os.path.join(self.run_path, "states")

		return resolved_datetime, index


t_start = 0


def t_step():
	"""
	returns delta t from last call of t_step()
	"""
	global t_start
	t_end = time.perf_counter()
	delta_t = t_end - t_start
	t_start = t_end
	return delta_t
