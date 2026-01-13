# STD2Vformer

A pytorch implementation for the paper: '*STD2Vformer: A prediction-length-agnostic model for Spatiotemporal Prediction*‘,  We reference the implementation of other baseline models from this [repository](https://github.com/TCCofWANG/Spatial-Temporal-Forecasting-Library).

# 🎯Overview

![](./image/overview.png)	

<center><p>Figure1.The overall architecture of the proposed STD2Vformer</p></center>				



# 📊Regular Prediction

![Regular Result](./image/Regular_Result.png)

Please refer to the paper for experimental results on additional datasets.

# 📊Flexible Prediction

![Flexible Result](./image/Flexible_Result.png)


# 📝Install dependecies

Install the required packages

```
pip install -r requirements.txt
```



# 👉Data Preparation

The Los Angeles traffic speed files (METR-LA), as well as the Los Angeles traffic flow files (PEMS04 and PEMS08), can be accessed and downloaded from [Baidu Yun](https://pan.baidu.com/s/1ShuACUFZGR0EnEkIoYSw-A?pwd=ib60) or [Google Drive](https://drive.google.com/drive/folders/1lcv-QYH7nAk9ciGFOurSam6SJVWaW-lg?usp=sharing). Please place these files in the `datasets/` folder.



# 🚀Run Experiment

We have provided all the experimental scripts for the benchmarks in the `./scripts` folder, which cover all the benchmarking experiments. To reproduce the results, you can run the following shell code.

```python
   ./scripts/train.sh
```

  

# 🌟Citation

If you find this work is helpful to your research, please consider citing our paper:

```
Comming Soon!
```

**Thanks for your interest in our work!**







