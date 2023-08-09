# TasaNet: temporal attention and spatial auxiliary network for moving Infrared dim and small target detection

## Model
<!-- <center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./img/model.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">TasaNet</div>
</center> -->
![TasaNet](./img/model.jpg)
<center>TasaNet</center>


## PR results on the DAUB and IRDST datasets
<!-- <center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./img/PR_1.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">DAUB</div>
</center> -->
![DAUB](./img/PR_1.jpg)

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./img/PR_2.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">IRDST</div>
</center>

## Datasets
### 1. [DUAB](https://www.scidb.cn/en/detail?dataSetId=720626420933459968) and [IRDST](https://xzbai.buaa.edu.cn/datasets.html)

### 2. The COCO format need to convert to txt format.
``` python 
python utils_coco/coco_to_txt.py
```
### 3. The class of dataset should write to a txt file. 
Such as model_data/classes.txt

## Train
The hyper-parameters are set in train.py
```python 
python tarin.py
```

## Evaluate
The hyper-parameters are set in vid_map_coco.py
```python 
python vid_map_coco.py
```

## Visualization
The hyper-parameters are setted in vid_predcit.py
```python 
python vid_predcit.py
```

## Reference
https://github.com/bubbliiiing/yolox-pytorch/