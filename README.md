# Code Generating Discrete Legendre Orthogonal Polynomials and the Legendre Delay Network Basis

Andreas Stöckel, December 2020

This repository contains the Python code accompanying the technical report “[Discrete Function Bases and Convolutional Neural Networks](https://arxiv.org/abs/2103.05609)”

### [📝 Read the code](dlop_ldn_function_bases/function_bases.py)

### [📓 Open the Jupyter Notebook](compare_bases.ipynb)

### [📒 Read the Technical Report](http://compneuro.uwaterloo.ca/files/publications/stoeckel.2021b.pdf)

## Usage

First of all, if you want to integrate parts of this code into your own project, feel free to just copy-paste the portionts of the code you'll need.

If you don't want to do this, simply install this package via `pip`. For example, run

```sh
pip3 install --user -e .
```

Depending on your environment, you may need to use `pip` instead of `pip3`. Also, if you're inside a virtual environment, you may have to skip the `--user` argument.

After installation, you can simply import the `dlop_ldn_function_bases` package into your Python script. For example, the following Python code will generate a DLOP basis with *q* = 6 and *N* = 20.
```python
import dlop_ldn_function_bases as bases

bases.mk_dlop_basis(q=6, N=20)
```

Passing the array returned by one of the `mk_*_basis` functions through `lowpass_filter_basis` will ensure the the incomding `N` samples are optimally low-pass filtered to be represented by *q* = 6 (Fourier) coefficients.

## Citation

```
Andreas Stöckel. Discrete function bases and convolutional neural networks.
arXiv preprint arXiv:2103.05609, 2021.
URL: https://arxiv.org/abs/2103.05609
```
```


## License

The code in this repository is licensed under the Creative Commons Zero license. To the extent possible under law, Andreas Stöckel has waived all copyright and related or neighboring rights to this code. Attribution is not required, but of course welcome. This work is published from: Canada.
