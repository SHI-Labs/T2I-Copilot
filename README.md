<div align="center">

# T2I-Copilot: A Training-Free Multi-Agent Text-to-Image System for Enhanced Prompt Interpretation and Interactive Generation (ICCV'25)

[Chieh-Yun Chen](https://chiehyunchen.github.io/), Min Shi, Gong Zhang, [Humphrey Shi](https://www.humphreyshi.com/home)


</div>

![Header Image Placeholder](assets/fig-Teaser-Mustang.jpg)

[![arXiv](https://img.shields.io/badge/arXiv-2507.20536-red)](https://arxiv.org/pdf/2507.20536) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/SHI-Labs/T2I-Copilot/blob/master/LICENSE)

## TL;DR
🚀 First T2I system to enable both pre- and post-generation user control

🤖 Propose a training-free multi-agent framework where 3 expert agents collaborate to boost interpretability & efficiency

🔍 Bridges human intent with AI creativity for truly interactive & controllable generation

💥 Matches top-tier models, Recraft V3, Imagen 3, and outperforms FLUX1.1-pro by +6.17% at just 16.6% of the cost

🥇 Beats FLUX.1-dev (+9.11%) and SD 3.5 Large (+6.36%) on GenAI-Bench with VQAScore

## 📰 News
- Jun. 26, 2025 | 🎉🎉🎉 T2I-Copilot is accepted by ICCV 2025.


<div align="center">
  <video src="place_holder" width="70%"> </video>
</div>

## 📑 Open-source Plan

 - [x] Inference codes
 - [x] Technical Report
 - [ ] Interactive Demo

## 🎨 Qualitative Performance
![Header Image Placeholder](assets/fig-qualitative_result.jpg)

## 📊 Quantitative Performance
![Header Image Placeholder](assets/fig-quantitative_result.png)

## 🚀 Quick Start

### Install & Run
```bash
# 1. Setup environment
conda env create -f environment.yaml
conda activate t2i_copilot

# 2. Put the OpenAI Key into .env
OPENAI_API_KEY=<YOUR_KEY_HERE>

# 3-1. Download checkpoint from PowerPaint (https://github.com/open-mmlab/PowerPaint): Image inpainting & editing
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/ models/PowerPaint/checkpoints/ppt-v2-1

# 3-2. Download checkpoint from GroundingSAM2 (https://github.com/IDEA-Research/Grounded-SAM-2): Mask generation for editing
cd models/Grounded_SAM2/checkpoints/
bash download_ckpts.sh
cd models/Grounded_SAM2/gdino_checkpoints/
bash download_ckpts.sh

# 4. Run with default settings (GPT-4o-mini, automatic mode)
python main.py --benchmark_name cool_sample.txt
```

## 🎯 System Modes

| Mode | Description | Command |
|------|-------------|---------|
| **Closed-source MLLM** (Default) & **Automatic** (Default) | GPT-4o-mini via OpenAI API | `python main.py` |
| **Open-source MLLM** | Mistral/Qwen via vLLM server | `python main.py --use_open_llm` |
| **Human-in-the-loop** | Interactive user feedback | `python main.py --human_in_the_loop` |

## 🔧 Open Source MLLM Setup

⚠️ **Use separate conda environment to avoid conflicts**

```bash
# Terminal 1: Start vLLM server
# Step i) 
conda env create -f env_vllm.yaml
conda activate vllm

# Step ii) Start server
# Option 1: Mistral
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --limit_mm_per_prompt 'image=10' --tensor-parallel-size 2

# Option 2: Qwen2.5-VL (not recommend 3B)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct

# Terminal 2: Run main.py (default model: Mistral-Samll-3.1-24B-Instruct-2503)
conda activate t2i_copilot
python main.py --use_open_llm
```

## 📋 Usage Examples

### Basic Usage
*Default using 2 L40S GPUs to load FLUX.1-dev and PowerPaint with its own GPU, correspondingly*
```bash
# GPT-4o-mini + Automatic mode
python main.py --benchmark_name GenAIBenchmark/genai_image_seed.json

# GPT-4o-mini + Human-in-the-loop
python main.py --human_in_the_loop --benchmark_name cool_sample.txt

# Open-source LLM + Human-in-the-loop
python main.py --use_open_llm --human_in_the_loop
```

### Advanced Usage
```bash
# Custom open-source model and port
python main.py --use_open_llm \
    --open_llm_model="Qwen/Qwen2.5-VL-7B-Instruct" \
    --open_llm_port=8080 \
    --human_in_the_loop
```

###  Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--benchmark_name` | Benchmark file path | `cool_sample.txt` |
| `--human_in_the_loop` | Enable interactive mode | `False` |
| `--use_open_llm` | Use open-source LLM | `False` |
| `--open_llm_model` | Open LLM model name | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |
| `--open_llm_port` | vLLM server port | `8000` |

## 🔍 Troubleshooting


- **PyTorch Error**: `RuntimeError: operator torchvision::nms does not exist`
    ```bash
    conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
    ```

- **vLLM Conflicts**: Use separate conda environments
    - vLLM server: `transformers>=4.51.1,<4.53.0`
    - Main.py: Different transformers version

## 📁 Output Structure

```
results/
├── GenAIBenchmark-fixseed/
│   ├── AgentSys_vRelease/
│   │   ├── 000_FLUX.1-dev.png
│   │   ├── 000_config.json
│   │   └── 000.log
│   └── AgentSys_vRelease_human_in_loop/
└── DrawBench/
```

## 🙏 Acknowledgements

We gratefully acknowledge the generous contributions of the open-source community, especially the teams behind **[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)**, **[PowerPaint](https://github.com/open-mmlab/PowerPaint)**, **[GroundingSAM2](https://github.com/IDEA-Research/Grounded-SAM-2)**, **[Mistral AI](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)**, **[QWen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)**, and **[vllm](https://docs.vllm.ai/en/latest/models/supported_models.html)**. Their publicly available code and models made this work possible.


## 📖 Citation

```bibtex
@inproceedings{t2i_copilot_2025,
  title={T2I-Copilot: A Training-Free Multi-Agent Text-to-Image System for Enhanced Prompt Interpretation and Interactive Generation},
  author={Chieh-Yun Chen and Min Shi and Gong Zhang and Humphrey Shi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

---

<div align="center">

**Made with ❤️ for the AI community**

</div> 