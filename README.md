# PRMG
PRMG is a deep-learning-based precipitation retrieval framework that employs multimodal fusion of infrared, passive microwave, and radar observations, together with geographical information correction, to address modality heterogeneity and terrain-induced biases.

For more information or paper **"Precipitation Retrieval integrating Multiple Satellite Observations: A Dataset and A Framework"**, please refer to [here](https://ieeexplore.ieee.org/abstract/document/11068950).


![image](https://github.com/Zjut-MultimediaPlus/PRMG/blob/main/img.png)
---

# The short introduction of files

**evaluate_test_two_stage_gpu.py**: Used for testing

**evaluation_index_gpu.py  && evaluate_train_gpu.py**: Testing auxiliary function libraries

**geo_channel.txt && geo_norm.txt**: Configuration files for the model, which use several geographical information data and altitude normalization parameters

**random_test_node3.txt**: List of the data reading order during training

**train_classify_geo_npy.py && train_regress_geo_npy.py**: Used for training precipitation identification models and precipitation value regression models

**pack_data_npy_v4.py**:Used to convert HDF format files to NPY format, in order to accelerate data reading during training

 

# Dateset

[[click to download Precipitation-MG]](https://pan.baidu.com/s/1Ciku8U78znWDX4ITD62TfA?pwd=9999)
fetch code: ```9999```

**PRMD_DATA_21-22_npy_v4**: Data file in NPY format, used to accelerate used to accelerate file reading during training

**PRMD_DATA_21-22_v4**: Data file in HDF format, convenient for viewing data with Panoply software

 

# Train

you can use **train_classify_geo_npy.py** and **train_regress_geo_npy.py** to train your new model.

 
# Test
[[click to download model]](https://pan.baidu.com/s/1-QbBTii8Ti2UYdn7MThGSw?pwd=9999)
fetch code: ```9999```

move the downloaded files into folder  ```checkpoints_cls``` and ```checkpoints_reg```  

You can use **evaluate_test_two_stage_gpu.py** to test your model.

 

# Environment
Python 3.8+ pytorch-1.12.0-cuda11.3-cudnn8.2 

# Citation
 ```
@ARTICLE{11068950,
  author={Wang, Zheng and He, Boxian and Wang, Chunjiao and Xu, Bin and Bai, Cong},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Precipitation Retrieval Integrating Multiple Satellite Observations: A Dataset and a Framework}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  keywords={Precipitation;Satellites;Spaceborne radar;Radar;Reflectivity;Clouds;Deep learning;Estimation;Data integration;Rain;Deep learning;geographic correction;multimodal data fusion;precipitation retrieval},
  doi={10.1109/TGRS.2025.3585407}} 
  ```
