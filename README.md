A deep learning-based ensemble method for helmet-wearing detection
====
demo code for paper "A deep learning-based ensemble method for helmet-wearing detection"


## Requirements

  * Tensorflow 1.12
  * Python 3.7

## Usage

```python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config```

```python export_inference_graph.py \ --input_type image_tensor \ --pipeline_config_path training/faster_rcnn_inception_v2_coco.config \  --trained_checkpoint_prefix training/model.ckpt-60000 \  --output_directory training/1```



## Cite

```
@article{fan2020deep,
  title={A deep learning-based ensemble method for helmet-wearing detection},
  author={Fan, Zheming and Peng, Chengbin and Dai, Licun and Cao, Feng and Qi, Jianyu and Hua, Wenyi},
  journal={PeerJ Computer Science},
  volume={6},
  pages={e311},
  year={2020},
  publisher={PeerJ Inc.}
}
```
