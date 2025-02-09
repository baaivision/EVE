## Data Composition

| Data name | Data size |
| --- | ---: |
| EVE_pretrain_cap33M.json | 28 GB |
| [LLaVA_v1_5_mix665K.json](https://drive.google.com/file/d/1cnuVSRQ6_W80ZBnEYCeCzG9KHMGO3XrG/view?usp=sharing) | 983 MB |
| [EVE_instruct_mix1.8M.json](https://drive.google.com/file/d/1iGg85xdJhyZv-s1ttCe_SZ-CUk-hThjs/view?usp=sharing) | 2.1 GB |

### EVE-PT Dataset
We introduce publicly available web-scale data, including image-only: SA-1B, OpenImages; and image-text: LAION. We remove noisy text captions and reproduce 33M high-quality descriptions via Emu2 (17B) and LLaVA-1.5 (13B) as EVE-cap33M. **We have no specific plan to release pretraining data.** You can download and filter images according to our paper's guidelines, utilizing [LLaVA-NEXT](https://github.com/LLaVA-VL/LLaVA-NeXT) to generate high-definition image descriptions, which would provide better results.

#### Prepare PT Images

Organize the data as follows in `./playground/data/EVE-Pretrain-33M/`:

```none
data
├── EVE-Pretrain-33M
│   │── eve_pretrain_cap33m.json
│   ├── LAION-Dedump
│   │   ├── images
│   │   │   ├── 000000
│   │   │   ├── 000001
│   │   │   ├── ...
│   ├── Openimages_v6
│   │   ├── images
│   │   │   ├── V6Train1
│   │   │   ├── V6Train2
│   │   │   ├── ...
│   ├── SAM-11M
│   │   ├── images
│   │   │   ├── 000000
│   │   │   ├── 000001
│   │   │   ├── ...
```


### EVE-SFT Dataset
We utilize LLaVA-v1_5-mix665K as SFT data to obtain the standard version of EVE-7B. Besides, we also attempt to enlarge the limitation of maximum resolution only in the SFT stage. To bridge the resolution gap between pre-training and fine-tuning stages, we further involve 1.2M SFT conversation data, including AI2D, Synthdog, DVQA, ChartQA, DocVQA, Vision-Flan, and Bunny-695K to obtain high-resolution version of EVE-7B-HD.

#### Prepare SFT Images

Please download the annotation of the final mixture SFT data: [llava_v1_5_mix665k.json](https://drive.google.com/file/d/1cnuVSRQ6_W80ZBnEYCeCzG9KHMGO3XrG/view?usp=sharing) and [eve_instruct_mix1.8m.json](https://drive.google.com/file/d/1iGg85xdJhyZv-s1ttCe_SZ-CUk-hThjs/view?usp=sharing); Then download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing). We save all files as `.jpg`
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [images](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [images2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- AI2D: [ai2d](https://huggingface.co/datasets/lmms-lab/ai2d)
- Synthdog: [synthdog-en](https://huggingface.co/datasets/naver-clova-ix/synthdog-en)
- DVQA: [DVQA](https://huggingface.co/datasets/skywalkerzhang19/DVQA)
- ChartQA: [ChartQA](https://huggingface.co/datasets/lmms-lab/ChartQA)
- DocVQA: [DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)
- Open_images: [Bunny-v1_0-data](https://huggingface.co/datasets/BoyaWu10/Bunny-v1_0-data)
- Vision-Flan: [vision-flan_191-task_1k](https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k)

Then, organize the data as follows in `./playground/data/EVE-Finetune/`:

```none
data
├── EVE-Finetune
│   │── llava_v1_5_mix665k.json
│   │── eve_instruct_mix1.8m.json
│   ├── ai2d
│   │   ├── images
│   │   ├── ...
│   ├── chartqa
│   │   ├── train
│   │   ├── val
│   │   ├── ...
│   ├── coco
│   │   ├── train2017
│   │   ├── ...
│   ├── docvqa
│   │   ├── train
│   │   ├── ...
│   ├── dvqa
│   │   ├── images
│   │   ├── ...
│   ├── gqa
│   │   ├── images
│   │   ├── ...
│   ├── ocr_vqa
│   │   ├── images
│   │   ├── ...
│   ├── open_images
│   │   ├── 0a0bc91825468c45.jpg
│   │   ├── ...
│   ├── syndog
│   │   ├── images
│   │   ├── ...
│   ├── textvqa
│   │   ├── train_images
│   │   ├── ...
│   ├── vg
│   │   ├── VG_100K
│   │   ├── VG_100K_2
│   │   ├── ...
│   ├── Vision-Flan_vision-flan_191-task_1k
│   │   ├── images_191task_1k
│   │   ├── ...
```
