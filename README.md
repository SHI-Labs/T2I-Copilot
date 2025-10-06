<div align="center">

# T2I-Copilot: A Training-Free Multi-Agent Text-to-Image System for Enhanced Prompt Interpretation and Interactive Generation (ICCV'25)

[Chieh-Yun Chen](https://chiehyunchen.github.io/), Min Shi, Gong Zhang, [Humphrey Shi](https://www.humphreyshi.com/home)

</div>


![Header Image Placeholder](assets/fig-Teaser-Mustang.jpg)

## 🎥 TL;DR

<div align="center">
  <video src="https://github.com/user-attachments/assets/75f7e356-b59a-4999-85f4-9a9e281e6570" width="100%"> </video>
</div>

🚀 First T2I system to enable both pre- and post-generation user control

🤖 Propose a training-free multi-agent framework where 3 expert agents collaborate to boost interpretability & efficiency

🔍 Bridges human intent with AI creativity for truly interactive & controllable generation

💥 Matches top-tier models, Recraft V3, Imagen 3, and outperforms FLUX1.1-pro by +6.17% at just 16.6% of the cost

🥇 Beats FLUX.1-dev (+9.11%) and SD 3.5 Large (+6.36%) on GenAI-Bench with VQAScore


## 📰 News
- Oct. 5, 2025 | Release gradio interactive demo
- Jun. 26, 2025 | 🎉🎉🎉 T2I-Copilot is accepted by ICCV 2025.


## 🎮 Gradio Demo
Default using 2 L40S GPUs to load FLUX.1-dev and PowerPaint with its own GPU, correspondingly
```bash
conda env create -f environment_demo.yaml
conda activate t2i_copilot_demo

# Download checkpoint from PowerPaint (https://github.com/open-mmlab/PowerPaint): Image inpainting & editing
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/ models/PowerPaint/checkpoints/ppt-v2-1

# Download checkpoint from GroundingSAM2 (https://github.com/IDEA-Research/Grounded-SAM-2): Mask generation for editing
cd models/Grounded_SAM2/checkpoints/
bash download_ckpts.sh
cd models/Grounded_SAM2/gdino_checkpoints/
bash download_ckpts.sh

python interactive_demo.py
```

<div align="center">
  <video src="https://github.com/user-attachments/assets/d86cc6a8-160e-4eb5-9b54-f7bb7e3e3da5" width="100%"> </video>
</div>

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