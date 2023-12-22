## Extractor and Profiler
### Environment Setup
#### (Option 1) Docker
```shell
sudo docker pull fenp/pytorch:1.12.1
sudo docker run --cpuset-cpus=0-$(($(nproc) - 1)) -v .:/app/AsyFunc -it --rm fenp/pytorch:1.12.1 /bin/bash
cd /app/AsyFunc/profiler_extractor
```

#### (Option 2) Anaconda or Pip
Please install the following software and packages

| software     | version |
| -----        | ----- |
| python       | 3.8  |
| pytorch      | 1.12 |
| torchvision  | 0.13 |
| transformers | 4.28 |
| ultralytics  | 8.0  |

### Extractor
#### Usage
usage: `python ./extractor.py <model_name> <core size> <batch size>`, the batch size should be 1 for extracting input data

example: `python ./extractor.py resnet50 4 1`

> There is a `depth` parameter in extractor.Model, which is to control the depth of model extraction. Since a layer can contain more sub-layers, a larger depth value will result in a deeper decomposition of the model and more layers (-1 indicates infinite depth). Therefore, different models require different depth parameters to avoid analyzing too many layers.

All example models have a default value for `depth`.

| Model Name      | Depth |
| ----------- | ----------- |
| Vgg16      | -1       |
| InceptionV3   | 2        |
| EfficientNet-b5   | 4        |
| SSD300   | -1        |
| yolov8x   | 4        |
| bert   | 5        |
| ViT_base   | -1        |
| resnet50   | -1        |

### Profiler
After extracting the model, use profiler to get layer metrics.
```shell
python ./profiler.py ./model_weight/<model_name>_layers/ <core size> <batch size>
```

or using shell to execute tests for all configurations
`nohup bash ./profiler.sh > log_$modelName.txt 2>&1 &`