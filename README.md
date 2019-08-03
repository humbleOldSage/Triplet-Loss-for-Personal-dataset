# Triplet-Loss-for-Personal-dataset

This work here is an addition to the novel work done by [omoindrot](https://github.com/omoindrot) in this [repo](https://github.com/omoindrot/tensorflow-triplet-loss). After improvement, the code lets one train and validate triplet loss for personal image dataset.

To train the triplet loss model on your dataset, type the following on terminal: <br  />
```
python3 train.py --model_dir experiments/batch_all --data_dir /Data/Train
```


The data should be in a folder "Data" in this format

<pre>
-Data
  |--Train
  |   |--Class1
  |        |--img1cls1.jpg
  |        |--img2cls1.jpg
  |        .
  |        .
  |   |--Class2<br />
  |        |--img1cls2.jpg
  |        |--img2cls2.jpg
  |        .
  |        .
  |--Test
      |--Class1
           |--img1cls1.jpg
           |--img2cls1.jpg
           .
           .
      |--Class2
           |--img1cls2.jpg
           |--img2cls2.jpg
           .
           .
             
</pre>

**This repository is still under work. As of 3rd August, only the train part works.The data for cross validation is taken care of from the Train data.**
