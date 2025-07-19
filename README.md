# Complex Neural Modules: A Type-Agnostic Module Set for Complex-Valued Machine Learning

Original Pytorch implementation for Complex Systems.

## About

This module set provides a general-purpose, type-agnostic framework for machine learning in the complex domain. 

This type-agnostic design makes the framework particularly suitable for large language models (LLMs) operating under quantization constraints or specialized hardware environments.

Designed to harness the unique structure of complex numbers—such as phase coherence and analytic behavior—it allows researchers and developers to easily extend models into the complex space without sacrificing flexibility or performance.

## Key features

The module set includes complex-valued implementations of commonly used components such as Linear, Conv2d, and Attention, enabling end-to-end training in the complex domain. 

***
### Formulation

```math
\displaylines{
x \in \mathbb{C^N} \\
W \in \mathbb{C^{M \times N}} \\
y = Wx
}
```

## Implementation and License

This repository is pure pytorch implementation.

Licensed under ["MIT License"](https://mit-license.org/).

Commercial use permitted

## How to use

- Clone the repository

```bash
git clone https://github.com/Rikka-Botan/Complex_Neural_Modules.git
```

- Model create

```python
from module.modeling import ComplexLinear, ComplexConv2d, ComplexAttention

cl = ComplexLinear(
  768, 768
)
cc = ComplexConv2d(
  5, 5
)
ca = ComplexAttention(
  768, 6
)
cl_output = cl(hidden_states)
cc_output = cc(hidden_states)
ca_output = ca(hidden_states)
```

## Acknowledgements

I thank the developers of python and pytorch.

I thank all the researchers for their efforts to date.

I thank Japan's high standard of education.

And most of all, thank you for your interest in this repository.

## Citations

I would be happy to include a citation at the end, but it is not required.

Feel free to use modules.


## Contact Us

[My X account](https://x.com/peony__snow)


## About Author

### Rikka Botan

Japanese independent researcher having shy and pampered personality >_<

Twin-tail hair is a charm point :)

Interested in natural language processings. 

Usually using python and C.

![RikkaBotan_Logo](https://github.com/user-attachments/assets/92913f91-9136-4d44-8b4d-8a2120118a05)
