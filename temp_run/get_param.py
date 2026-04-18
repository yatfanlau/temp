import argparse


params = argparse.Namespace()


def str2bool(v):
	"""
	'type variable' for add_argument
	"""
	if isinstance(v, bool):
		return v
	if v.lower() in ("yes", "true", "t", "y", "1"):
		return True
	if v.lower() in ("no", "false", "f", "n", "0"):
		return False
	raise argparse.ArgumentTypeError("boolean value expected.")


def build_parser():
	"""
	Build the fluid-only CLI parser used across training and evaluation scripts.
	"""
	parser = argparse.ArgumentParser(description="train / test a pytorch model to predict fluid states")

	# Training parameters
	parser.add_argument("--n_epochs", default=100000, type=int, help="number of epochs (after each epoch, the model gets saved)")
	parser.add_argument("--n_batches_per_epoch", default=10000, type=int, help="number of batches per epoch (default: 10000)")
	parser.add_argument("--batch_size", default=50, type=int, help="batch size (default: 50)")
	parser.add_argument("--n_samples", default=10, type=int, help="number of samples (different offsets) per batch (default: 10)")
	parser.add_argument("--dataset_size", default=1000, type=int, help="size of dataset (default: 1000)")
	parser.add_argument("--average_sequence_length", default=5000, type=int, help="average sequence length in dataset (default: 5000)")
	parser.add_argument("--resolution_factor", default=8, type=int, help="resolution factor for superres / kernels (default: 8)")
	parser.add_argument("--loss_bound", default=20, type=float, help="loss factor for boundary conditions")
	parser.add_argument("--loss_domain_res", default=10, type=float, help="loss factor for residual loss in fluid domain")
	parser.add_argument("--loss_domain_up", default=0, type=float, help='loss factor for "upwind loss" in fluid domain')
	parser.add_argument("--loss_domain_p", default=0, type=float, help="loss factor for pressure in fluid domain")
	parser.add_argument("--loss_mean_a", default=0, type=float, help="unused legacy argument kept for compatibility")
	parser.add_argument("--loss_mean_p", default=0, type=float, help="unused legacy argument kept for compatibility")
	parser.add_argument("--border_weight", default=0, type=float, help="extra weight on fluid domain borders")
	parser.add_argument("--lr", default=0.0001, type=float, help="learning rate of ADAM-optimizer (default: 0.0001)")
	parser.add_argument("--clip_grad_norm", default=None, type=float, help="gradient norm clipping (default: None)")
	parser.add_argument("--clip_grad_value", default=None, type=float, help="gradient value clipping (default: None)")
	parser.add_argument("--cuda", default=True, type=str2bool, help="use GPU")
	parser.add_argument("--log_loss", default=True, type=str2bool, help='logarithmic loss to "normalize" gradients')
	parser.add_argument("--log_every", default=10, type=int, help="log scalar metrics every N batches")

	# Network parameters
	parser.add_argument("--net", default="Fluid_model", type=str, help="network to train", choices=["Fluid_model"])
	parser.add_argument("--hidden_size", default=20, type=int, help="hidden size of network (default: 20)")
	parser.add_argument("--orders_v", default=2, type=int, help="spline order for velocity potential field (default: 2)")
	parser.add_argument("--orders_p", default=2, type=int, help="spline order for pressure field (default: 2)")

	# Fluid parameters
	parser.add_argument("--rho", default=1, type=float, help="fluid density rho")
	parser.add_argument("--mu", default=1, type=float, help="fluid viscosity mu")
	parser.add_argument("--dt", default=1, type=float, help="dt per time integration step")

	# Domain parameters
	parser.add_argument("--width", default=None, type=int, help="domain width")
	parser.add_argument("--height", default=None, type=int, help="domain height")
	parser.add_argument("--max_speed", default=1, type=float, help="max speed for boundary conditions in dataset (default: 1)")

	# Logger / Load parameters
	parser.add_argument("--plot", default=False, type=str2bool, help="legacy argument kept for compatibility")
	parser.add_argument("--log", default=True, type=str2bool, help="save metrics / checkpoints and write tensorboard logs")
	parser.add_argument("--run_name", default=None, type=str, help="optional run name; default uses the fluid hyperparameter string")
	parser.add_argument("--run_path", default=None, type=str, help="load a copied run directly from this exact directory")
	parser.add_argument("--load_date_time", default=None, type=str, help="date_time of run to load (default: None)")
	parser.add_argument("--load_index", default=None, type=int, help="index of run to load (default: latest state in the selected run)")
	parser.add_argument("--n_warmup_steps", default=None, type=int, help="number of warm up steps to perform when loading a pretrained model")
	parser.add_argument("--load_optimizer", default=False, type=str2bool, help="load state of optimizer (default: False)")
	parser.add_argument("--load_latest", default=False, type=str2bool, help="load latest version for training (if True: leave load_date_time and load_index None)")
	return parser


def parse_params(args=None, allow_unknown=True):
	parser = build_parser()
	if allow_unknown:
		return parser.parse_known_args(args=args)[0]
	return parser.parse_args(args=args)


def set_params(values):
	if isinstance(values, argparse.Namespace):
		values = dict(vars(values))
	else:
		values = dict(values)
	params.__dict__.clear()
	params.__dict__.update(values)
	return params


def apply_fluid_defaults(namespace):
	if namespace.width is None:
		namespace.width = 230
	if namespace.height is None:
		namespace.height = 49
	if getattr(namespace, "net", None) is None:
		namespace.net = "Fluid_model"
	return namespace


def print_params(namespace=None):
	namespace = params if namespace is None else namespace
	print(f"Parameters: {vars(namespace)}")


def get_hyperparam_fluid(namespace):
	return f"fluid net {namespace.net}; hs {namespace.hidden_size}; ov {namespace.orders_v}; op {namespace.orders_p}; mu {namespace.mu}; rho {namespace.rho}; dt {namespace.dt};"


def get_run_name(namespace):
	return namespace.run_name or get_hyperparam_fluid(namespace)


def toCuda(x):
	if isinstance(x, (tuple, list)):
		return [toCuda(xi) if params.cuda else xi for xi in x]
	return x.cuda() if params.cuda else x


def toCpu(x):
	if isinstance(x, (tuple, list)):
		return [toCpu(xi) for xi in x]
	return x.detach().cpu()


set_params(apply_fluid_defaults(parse_params(allow_unknown=True)))
