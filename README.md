# STD2Vformer



# Overview

![](./image/overview.png)	

<center><p>Figure1.The overall architecture of the proposed STD2Vformer</p></center>				



# Regular Prediction

![Regular Result](/image/Regular Result.png)

# Flixible Prediction

![Flexible Result](/image/Flexible Result1.png)

![Flexible Result](/image/Flexible Result2.png)



## Data Preparation

The Los Angeles traffic speed files (METR-LA), as well as the Los Angeles traffic flow files (PEMS04 and PEMS08), can be accessed and downloaded from [Baidu Yun](https://pan.baidu.com/s/1ShuACUFZGR0EnEkIoYSw-A?pwd=ib60). Please place these files in the `datasets/` folder.



## Get Started

We have provided all the experimental scripts for the benchmarks in the `./scripts` folder, which cover all the benchmarking experiments. To reproduce the results, you can run the following shell code.

```python
   ./scripts/train.sh
```



