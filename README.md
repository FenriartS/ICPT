<!-- ## Train -->
### Preparation
Download pretrained VQGAN codebook checkpoint and config [_vqgan_imagenet_f16_1024_](https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/?p=%2F), place both _last.ckpt_ and _model.yaml_ on the repository root. 

Download pretrained MAE-VQGAN [weights](https://drive.google.com/file/d/130vNSlqg3faHzVGGh_vkeUUh-2uVX3se/view?usp=sharing) to ./pretrained_model/

### Dataset
Prepare PASCAL-5i dataset in [Volumetric Aggregation Transformer](https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer) for foreground segmentation. Download [ImageNet](https://www.image-net.org/), [NYUDepthv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), [Synthetic Rain Datasets](https://paperswithcode.com/dataset/synthetic-rain-datasets), [LoL](https://paperswithcode.com/dataset/synthetic-rain-datasets) for colorization, depth estimation, deraining, enhancement to ./datasets/ , respectively.

### Training
```
python main_tuning.py --output_dir /output/path
```

### Inference
On foreground segmentation:
```
cd evaluate &&
python evaluate_segmentation.py --prompt_ckpt /path/to/prompt
```
On colorization:
```
cd evaluate &&
python evaluate_colorization.py --prompt_ckpt /path/to/prompt
```
