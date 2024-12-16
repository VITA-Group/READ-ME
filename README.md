# Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design
[Ruisi Cai](https://cairuisi.github.io/)<sup>1</sup>,
[Yeonju Ro](https://sites.google.com/view/hey-yeonju)<sup>1</sup>,
[Geon-Woo Kim](https://gwsshs22.github.io/)<sup>1</sup>,
[Peihao Wang](https://peihaowang.github.io/)<sup>1</sup>,
[Babak Ehteshami Bejnordi](https://babakint.com/)<sup>2</sup>,
[Aditya Akella](https://www.cs.utexas.edu/~akella/)<sup>1</sup>,
[Zhangyang Wang](https://vita-group.github.io/)<sup>1</sup>

<sup>1</sup>University of Texas at Austin, <sup>2</sup>Qualcomm AI Research

## Usage 
The code is based on the Hugging Face Transformers repository. We modified `src/transformers/model/modeling_llama.py` to integrate the MoE-fication process.

The main scripts are located in the moefication directory. Start by running the preprocessing scripts, `moefication/scripts/preprocess_1.sh` and `moefication/scripts/preprocess_2.sh`, to generate experts. After preprocessing, train the model using moefication/scripts/train.sh.

## Citation
If you find this useful, please cite the following paper:
```
@inproceedings{
cai2024textitreadme,
title={\${\textbackslash}textit\{Read-{ME}\}\$: Refactorizing {LLM}s as Router-Decoupled Mixture of Experts with System Co-Design},
author={Ruisi Cai and Yeonju Ro and Geon-Woo Kim and Peihao Wang and Babak Ehteshami Bejnordi and Aditya Akella and Zhangyang Wang},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=i8JaxY7tDI}
}
```