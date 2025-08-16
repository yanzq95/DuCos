<p align="center">
<h3 align="center"> DuCos: Duality Constrained Depth Super-Resolution via Foundation Model
<br>
:star2: ICCV 2025 :star2:
</h3>

<p align="center">
<a href="https://yanzq95.github.io/">Zhiqiang Yan</a><sup>1</sup>, 
<a href="https://scholar.google.com/citations?user=VogTuQkAAAAJ&hl=zh-CN">Zhengxue Wang</a><sup>2</sup>, 
<a href="">Haoye Dong</a><sup>1</sup>,
<a href="">Jun Li</a><sup>2</sup>,
<a href="https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN">Jian Yang</a><sup>2</sup>,
<a href="">Gim Hee Lee </a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>National University of Singapore&nbsp;&nbsp;&nbsp;
  <br>
  <sup>2</sup>Nanjing University of Science and Technology&nbsp;&nbsp;&nbsp;
</p>

## Dependencies
Please refer to the <a href="https://github.com/yanzq95/DuCos/blob/main/requirements.sh">requirements</a> file.

## Datasets

[RGB-D-D](https://github.com/lingzhi96/RGB-D-D-Dataset)

[TOFDSR](https://yanzq95.github.io/projectpage/TOFDC/index.html)

[NYU-v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

[Lu & Middlebury](https://web.cecs.pdx.edu/~fliu/project/depth-enhance/)

[Hypersim](http://github.com/apple/ml-hypersim)

## Models

Pretrained models can be found in  <a href="">checkpoints</a>.

## Training

```
python main.py --scale 4 --model_name DuCos --save DuCos_X4 --gpus 0

python main_float.py --model_name DuCos --save DuCos_X2.7 --gpus 0

python main_compress.py --model_name DuCos --save DuCos_compress --gpus 0

python main_RGBDD.py --scale 4 --model_name DuCos --isNosiy 0 --save DuCos_RGBDD --gpus 0

python main_TOFDSR.py --scale 4 --model_name DuCos --isNosiy 0 --save DuCos_TOFDSR --gpus 0
```

## Testing

```
python test_Sync.py --scale 4 --model_name DuCos --pretrain "./ckpts/x4.pth.tar" --gpu 0

python test_Sync_float.py --model_name DuCos --pretrain "./ckpts/x2.7.pth.tar" --gpu 0

python test_Sync_compress.py --model_name DuCos --pretrain "./ckpts/Compressx8.pth.tar" --gpu 0

python test_RealRGBDD.py --scale 4 --model_name DuCos --isNosiy 0 --pretrain "./ckpts/RealRGBDD.pth.tar" --gpu 0

python test_RealTOFDSR.py --scale 4 --model_name DuCos --isNosiy 0 --pretrain "./ckpts/RealTOFDSR.pth.tar" --gpu 0
```

## Citation

If our method proves to be of any assistance, please consider citing:
```
@article{yan2025ducos,
  title={DuCos: Duality Constrained Depth Super-Resolution via Foundation Model},
  author={Yan, Zhiqiang and Wang, Zhengxue and Dong, Haoye and Li, Jun and Yang, Jian and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2503.04171},
  year={2025}
}
```
