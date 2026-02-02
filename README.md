<div align="center">
  <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap; text-align: left;">
      AdaptMMBench: Benchmarking Adaptive Multimodal Reasoning for Mode Selection and Reasoning Process
    </h1>
  </div>


  <br>


  <a href="https://arxiv.org/abs/2505.15436v2">
    <img src="https://img.shields.io/badge/ArXiv-CoF-brown?logo=arxiv" alt="Paper">
  </a>
  <a href="https://adaptmmbench.github.io/">
    <img src="https://img.shields.io/badge/-HomePage-black?logo=github" alt="ProjectPage">
  </a>
  <a href="https://huggingface.co/datasets/xintongzhang/CoF-SFT-Data-5.4k">
    <img src="https://img.shields.io/badge/🤗 huggingface-Dataset-purple" alt="Dataset">
  </a>
</div>


<br>
<span>
<b>Authors:</b> 
<a class="name" target="_blank" href="https://github.com/xtong-zhang">Xintong Zhang<sup>*</sup></a>, 
<a class="name" target="_blank" href="https://zhigao2017.github.io/">Zhi Gao<sup>*</sup></a>, 
<a class="name" target="_blank" href="https://bofei5675.github.io/">Bofei Zhang</a>, 
<a class="name" target="_blank" href="https://pengxiang-li.github.io/">Pengxiang Li</a>, 
<a class="name" target="_blank" href="https://adatwi.github.io/">Xiaowen Zhang</a>, 
<a class="name" target="_blank" href="https://adatwi.github.io/">Yang Liu</a>, 
<a class="name" target="_blank" href="https://adatwi.github.io/">Tao Yuan</a>, 
<a class="name" target="_blank" href="https://wu-yuwei-bit.github.io/">Yuwei Wu<sup>†</sup></a>, 
<a class="name" target="_blank" href="https://scholar.google.com/citations?user=Sl6TV7gAAAAJ&hl=en">Yunde Jia</a>, 
<a class="name" target="_blank" href="https://www.zhusongchun.net/">Song-Chun Zhu</a>, 
<a class="name" target="_blank" href="https://liqing.io/">Qing Li<sup>†</sup></a>
<br>
<sup>*</sup>Equal Contribution. 
<sup>†</sup>Corresponding Author.
</span>


# 🔥News
- [2026/02/03] We released our RL dataset, model, training code, welcome to download and explore them.
welcome!

<br>

# Overview

![overview](./assets/teaser.jpg)

<details><summary>Abstract</summary> 
Adaptive multimodal reasoning has emerged as a promising frontier in Vision-Language Models (VLMs), aiming to dynamically modulate between tool-augmented visual reasoning and text reasoning to enhance both effectiveness and efficiency. However, existing evaluations rely on static difficulty labels and simplistic metrics, which fail to capture the dynamic nature of difficulty relative to varying model capacities. Consequently, they obscure the distinction between adaptive mode selection and general performance while neglecting fine-grained process analyses. In this paper, we propose AdaptMMBench, a comprehensive benchmark for adaptive multimodal reasoning across five domains: real-world, OCR, GUI, knowledge, and math, encompassing both direct perception and complex reasoning tasks. AdaptMMBench utilizes a Matthews Correlation Coefficient (MCC) metric to evaluate the selection rationality of different reasoning modes, isolating this meta-cognition ability by dynamically identifying task difficulties based on models' capability boundaries. Moreover, AdaptMMBench facilitates multi-dimensional process evaluation across key step coverage, tool effectiveness, and computational efficiency. Our evaluation reveals that while adaptive mode selection scales with model capacity, it notably decouples from final accuracy. Conversely, key step coverage aligns with performance, though tool effectiveness remains highly inconsistent across model architectures.
</details>

## Visual Search Agent
![visual_search_agent](./assets/visual_agent.jpg)

## Framework
![framework](./assets/model_inference.jpg)

<br>

# Training

## SFT Stage

### Installation

Please follow the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository to install the environment.

### Data Preparation

1. Download the dataset (including images and annotations) from [Hugging Face – Cof SFT Dataset](https://huggingface.co/datasets/xintongzhang/CoF-SFT-Data-5.4k)

2. Modify the configuration file `configs/sft_lora-7b.yaml` to match your data paths and training settings.
3. Copy `configs/dataset_info.json` to your image folder.

### Launch Training

Training can be started with the following script.

```bash
conda activate llamafactory
bash ./slurm_jobs/sft/train_7b_lora.sh
```

## RL Stage

### Installation

Please follow the [verl](https://github.com/volcengine/verl) repository to install the environment.

### Data Preparation

Download the dataset (including images and annotations) from [Hugging Face – CoF RL Dataset](https://huggingface.co/datasets/xintongzhang/CoF-RL-Data)


### Launch Training

Training can be started with the following script.

```bash
conda activate verl
cd ./verl
bash ./slurm_jobs/rl/run.sh
```

<br>

# Evaluation

### Installation

Set up an environment with `vllm`.

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm==0.8.2
```

### Prepare Data and Model

The Vstar Benchmark serves as an example dataset and can be downloaded from [Vstar benchmark](https://huggingface.co/datasets/craigwu/vstar_bench).

The model can be downloaded from [Hugging Face – Cof SFT Model 7B](https://huggingface.co/xintongzhang/CoF-sft-model-7b).

### Inference

Run the inference script using vllm.

```bash
conda activate vllm
bash ./slurm_jobs/eval/inference_vstar.sh
```


### Performance Metrics

To evaluate the model's performance on the VSTAR benchmark, begin by launching a dedicated vllm server process to serve the evaluation model (e.g., a judge model):

```bash
vllm serve /path/to/Qwen2.5-VL-72B-Instruct \
    --served-model-name judge \
    --port 51232 \
    --limit-mm-per-prompt image=1 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --disable-log-requests
```

Once the vllm service is running, execute the evaluation script to compute metrics:

```bash
bash ./slurm_jobs/eval/metrics_vstar.sh
```


### Adaptive Performance

**🔍 Resolution-Aware Zooming:** Adaptive-CoF dynamically adjusts its zoom-in scope, tailoring its focus based on image clarity and resolution.
<div align="center">
<img src="./assets/adaptive_case_1.png" width="90%" />
</div>

**🔄 Efficiency-First Strategy:** As image resolution improves, Adaptive-CoF automatically transitions from *iterative zooming* to *direct observation*, optimizing for both accuracy and speed.
<div align="center">
<img src="./assets/adaptive_case_2.png" width="90%" />
</div>

**🧠 Adaptive Reasoning in Action:** Adaptive-CoF flexibly shifts its strategy based on task difficulty:
- **Simple Tasks:** Solved directly without zooming (Top).
- **Moderate Tasks:** Resolved with a single targeted zoom (Middle).
- **Complex Tasks:** Requires iterative visual search for fine-grained details (Bottom).
<div align="center">
<img src="./assets/adaptive_case_3.png" width="90%" />
</div>


<!-- ## Citation
If you find our project helpful, please consider citing it using the following reference:
```bibtex
@article{zhang2025chain,
      title={Adaptive Chain-of-Focus Reasoning via Dynamic Visual Search and Zooming for Efficient VLMs},
      author = {Zhang, Xintong and Gao, Zhi and Zhang, Bofei and Li, Pengxiang and Zhang, Xiaowen and Liu, Yang and Yuan, Tao and Wu, Yuwei and Jia, Yunde and Zhu, Song-Chun and Qing Li},
      journal={arXiv preprint arXiv:2505.15436},
      year={2025}
}
``` -->
