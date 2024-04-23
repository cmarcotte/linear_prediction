# linear_prediction
Simple Python code and shell driver for the prediction of critical quenching perturbations.

Includes example data (from the FitzHugh-Nagumo model with $\gamma = 0.001$ on a ring of length $L=300$) and a `run.sh`[^1] script.

Figures for input, testing, and progress are outputted during execution (`./run.sh`) into `$savedir`.

Requires (see `requirements.txt`, `python -m venv` is recommended):
- python (3.11.0)
	- docopt (0.6.2)
	- numpy (1.24.2)
	- scipy (1.10.1)
	- h5py (3.8.0)
	- matplotlib (3.7.1)

Optionally, `ffmpeg` may be used to create an animation of the continuation process by uncommenting line 25 in `run.sh`.

[^1]: `run.sh` is written for `zsh`; minor edits may be needed for other shells.