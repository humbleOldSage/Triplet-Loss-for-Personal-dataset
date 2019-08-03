# Triplet-Loss-for-Personal-dataset

This work here is an addition to the novel work done by [omoindrot](https://github.com/omoindrot). A clear idea about triplet loss is given in the official [repo](https://github.com/omoindrot/tensorflow-triplet-loss). After improvement, the code lets one train and validate triplet loss for **ones own image dataset**.
The data pipelining has been updated for the code to work .

To train the triplet loss model on your dataset, type the following on terminal: <br  />
**for batch all technique**
```
python3 train.py --model_dir experiments/batch_all --data_dir /Data/Train
```
<br  />
OR 
<br  />
**for batch hard technique**
```
python3 train.py --model_dir experiments/batch_hard --data_dir /Data/Train
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

**This repository is still under work.**<br  />
  
  **UPDATE**<br  />
  
  `As of 3rd August,2019 only the train part works.The data for cross validation is taken care of from the Train data.`

<dl>
  <dt>UPCOMING</dt>
  <dd>A tutorial explaining code flow</dd>
  
  <dd>Test function</dd>
 
</dl>
