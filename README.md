# SELD


### How to Train This model

1. Download Dataset from https://zenodo.org/record/5476980 
(make sure the dataset version is 1.2 as this is the version that provides metadata for the evaluation dataset)

2. Preprocess Data with batch_feature_extracion.py in https://github.com/sharathadavanne/seld-dcase2021
the extracted features and labels should be in folders with these names:
seld_features_labels/DCASE2021/feat_label


in my experience the feature extractor in the code is not working properly and the code is better used with the above extractor.
(since the batch feature extractor linked above was made before the dcase competition it doesn't encounter for the metadata of the evaluation dataset 
(metadata_eval) that's why I suggest running only dev mode in the batch_feature_extracion.py line 6 should be :

```python
process_str = 'dev'
```

and I suggest extracting features twice once for the development dataset as usual
and once where you rename the metadata_eval to metadata_dev and the folder inside it to dev_test
then use the extractor a second time. this way you'll be able to extract labels and features for both the development and the evaluation dataset.)

3. create a new conda environment with the requirements.txt file

4. run params.py to generate a config files with the params of your choice. (the default are the params I used.) make sure a config file with the same name does not already exist.

5. run trainv2.py with arguments. You should set 'abspath' to the directory preprocessed datasets are located.

You should only train on the development dataset using the splits included in the code. and not the evaluation dataset. 

### My Contribution and how to disable it.
The code belongs to the original authors and I only changed minor things to get better results and or better training performance.
I made the model utilize my GPU instead of only cpu which lead to a much faster training time 
however you can change that by changing the 
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```
to 
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

in the script you that you want to not use the GPU.

I added cutout data augmentation to the dataset doubling the dataset size
the logic of the cutout algorithm can be found in the file:
[extra_augmentations.py](https://github.com/Abdelrahman-7gab/SELD/blob/mixup-trials/extra_augmentations.py)

as I believed that since the model was perfoming better on the train splits 
than the val and test splits that it's overfitting to some extent and started to memorize the train set.
this resulted in 4-5% better scores on my machine.
this change is apparent in data_loader.py in the next chunk of code:
        
```python
  if(extras):
        print("extra augmentation is on")
        cutoutDS = dataset.map(eraser, num_parallel_calls=AUTOTUNE, deterministic=deterministic)
        dataset = dataset.concatenate(cutoutDS)
        dataset = dataset.shuffle(2400)
```
        
which you can comment out to remove the extra cutout augmentation.
or you can also change line 157 in trainv2.py from 
```python
extras= mode == 'train'
```
to
```python
extras= false
```

and this should give you the same result.

### How to compute results.
use make_answer.py after training your model after changing the path to the location of your trained model.
make sure to select either development or evaluation dataset in the code before running the file.
this will output predections in the format of csv in the output_path of your choice.

after outputting all the predictions copy them to the results folder in https://github.com/sharathadavanne/seld-dcase2021
you can then run cls_compute_seld_results.py which will give you accurate results.

make sure you are using the correct metadata for the csv files that you're using. 

all rights for the original authors:
### (C) 2021. lsg1213, daniel03c1, ParkJinHyeock all rights reserved.
