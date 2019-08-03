import tensorflow as tf
import os,sys
import pathlib

def _ondisk_parse_(filename):
    img_raw = tf.io.read_file(filename)
    img_tensor = tf.image.decode_jpeg(img_raw,channels=3)
    img_tensor = tf.cast(img_tensor,tf.int8)
    img_tensor = tf.cast(img_tensor,tf.float32)
    img_rshpd = tf.image.resize(img_tensor,[128,128])
    #img_final = img_rshpd/255.0             NOT SURE ABOUT THIS PART 
    return img_rshpd

def train_input_fn(filenames,params,data_root):
    path_ds  = tf.data.Dataset.from_tensor_slices(filenames)
   # dataset = dataset.map(lambda x:_ondisk_parse_(x)).shuffle(params.train_size).batch(params.batch_size)
    img_ds= path_ds.map(_ondisk_parse_)
    #img_ds = path_ds.map(_ondisk_parse_,num_parallel_calls=AUTOTUNE)

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in filenames]
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    img_l_ds = tf.data.Dataset.zip((img_ds, label_ds))
    #ds = img_l_ds.cache(filename='./cache.tf-data')
    ds= img_l_ds.shuffle(params.train_size)
    ds = ds.repeat(params.num_epochs)
    ds = ds.batch(params.batch_size).prefetch(1)


    print("*******",ds)
    return ds


#def test_input_fn(filenames,params):
