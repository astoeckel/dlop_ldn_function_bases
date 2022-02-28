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

Andreas Stöckel. Discrete function bases and convolutional neural networks.  
arXiv preprint arXiv:2103.05609, 2021.  
URL: https://arxiv.org/abs/2103.05609


## License

The code in this repository is licensed under the Creative Commons Zero license. To the extent possible under law, Andreas Stöckel has waived all copyright and related or neighboring rights to this code. Attribution is not required, but of course welcome. This work is published from: Canada.

## Patent notice

Parts of this work related to the Legendre Delay Network, Legendre Memory Unit, or feed-forward Legendre Memory Unit are protected by copyright, patents and/or provisional patents owned by Applied Brain Research Inc. (ABR; see https://appliedbrainresearch.com/). Relevant patents include, but are not limited to:

EP3796229A1, US11238337B2, CA2939561A1:
"Methods and systems for implementing dynamic neural networks"

US20210342668A1:
"Methods And Systems For Efficient Processing Of Recurrent Neural Networks"

US11238345B2:
"Legendre memory units in recurrent neural networks"

63/313,676:
“Efficient Linear Systems for Neural Network Applications”

The author of this repository (Andreas Stöckel) believes that the code in this repository only implements individual components, but not the complete systems described in these patents.

Still, any non-exclusive patent rights potentially issued through the license of this repository do not apply to patents owned by ABR, even if the author may be listed as an inventor or co-inventor in current or future patents owned by ABR.

Note that ABR grants a non-exclusive license to these patents for non-commercial or academic use. You can obtain a free license, or buy a license for commercial use at the following URL: https://appliedbrainresearch.com/store/

