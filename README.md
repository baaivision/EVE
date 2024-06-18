# <img src="images/eve_logo.png" style="vertical-align: -10px;" :height="30px" width="30px"> EVE: Unveiling Encoder-Free Vision-Language Models

🚀🚀🚀 Official pytorch implementation of **[Unveiling Encoder-Free Vision-Language Models](http://arxiv.org/abs/2406.11832)**.

## 📜 News
We'll release the code and models within a week ! 💥

## 🤔 Motivation
<p align="center">
  <img src="images/eve_motivation1.png">
</p>

<p align="center">
  <img src="images/eve_structure.png">
</p>

- **Authors**: [Haiwen Diao*](https://scholar.google.com/citations?user=46eCjHQAAAAJ&hl=zh-CN), [Yufeng Cui*](https://scholar.google.com/citations?user=5Ydha2EAAAAJ&hl=zh-CN&oi=ao), [Xiaotong Li](https://scholar.google.com/citations?hl=zh-CN&user=cpCE_T4AAAAJ), [Yueze Wang](https://openreview.net/profile?id=~Yueze_Wang1), [Huchuan Lu📧](https://scholar.google.com/citations?user=D3nE0agAAAAJ&hl=zh-CN), [Xinlong Wang📧](https://scholar.google.com/citations?user=DPz0DjYAAAAJ&hl=zh-CN)

- **Institutes**: Dalian University of Technology; Beijing Academy of Artificial Intelligence; Peking University

## 💡 Highlights
- 🔥 **Superior Capability:** *Originated encoder-free* LVLM with *arbitrary* image aspect ratio, outperforming the counterpart *[Fuyu-8B](https://huggingface.co/adept/fuyu-8b)* and approaching *modular encoder-based* LVLMs.  

- 🔥 **Data Efficiency:** Filter solely *33M* publicly avaliable data from OpenImages, SAM, LAION for pre-training; Utilizing *665K* LLaVA SFT data for EVE-7B, and extra *1.2M* SFT data for EVE-7B (HD).  

- 🔥 **Training Efficiency:** Trained with *two 8-A100 (40G) nodes* in ~*9* days.  

- 🔥 **Pioneering Route:** An *efficient*, *transparent*, and *practical* strategy for developing pure decoder-only architecture across modalities.  

## 🤖 Model Zoo

| Model | LLM | Weight | VQAv2 | GQA | VizWiz | SQA_I | TextVQA | POPE | MME_P | MMBench | MM_Vet | 
|---|---|---|---|---|---|---|---|---|---|---|---|
| EVE_7B | Vicuna_7B | [checkpoint] | 75.4 | 60.8 | 41.8 | 63.0 | 51.9 | 83.6 | 1217.3 | 49.5 | 25.6 |
| EVE_7B_HD | Vicuna-7B | [checkpoint] | 78.6 | 62.6 | 51.1 | 64.9 | 56.8 | 85.0 | 1305.7 | 52.3 | 25.7 |

## License
The content of this project itself is licensed under [LICENSE](https://github.com/baaivision/EVE/blob/main/LICENSE).


## ✒️ Citation
If **EVE** is helpful for your research, please consider **star** ⭐ and **citation** 📝 :
```bibtex
@article{diao2024EVE,
  title={Unveiling Encoder-Free Vision-Language Models},
  author={Diao, Haiwen and Cui, Yufeng and Li, Xiaotong and Wang, Yueze and Lu, Huchuan and Wang, Xinlong},
  journal={arXiv preprint arXiv:2406.11832},
  year={2024}
}
```

## License
The content of this project itself is licensed under [LICENSE](https://github.com/baaivision/EVE/blob/main/LICENSE).