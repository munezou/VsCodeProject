# tf.data Performance

Note:     .           

[  ](https://github.com/tensorflow/docs/blob/master/site/en/guide/data_performance.md)
    .     
[tensorflow/docs](https://github.com/tensorflow/docs)     
.    
[docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
  .

## 

GPU TPU          .        
      .`tf.data` API      
.            `tf.data` API  .

    :

*      
    [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load)  
    .
*           .
*         .

##   

     ETL    :

1.  ****: (NumPy) (HDD  SSD)  (
    [GCS](https://cloud.google.com/storage/)
    [HDFS](https://en.wikipedia.org/wiki/Apache_Hadoop#Hadoop_distributed_file_system))
       .
2.  ****:   CPU  (shuffling), (batching),    
    (augmentation),              
    .
3.  ****:   () . (,    GPU TPU)

  CPU        .    ETL   
       .

     TFRecord      -  (batch)  
   .   `tf.data.Dataset`  `tf.keras`   
API   .

```
def parse_fn(example):
  "TFExample      ."
  example_fmt = {
    "image": tf.FixedLengthFeature((), tf.string, ""),
    "label": tf.FixedLengthFeature((), tf.int64, -1)
  }
  parsed = tf.parse_single_example(example, example_fmt)
  image = tf.io.image.decode_image(parsed["image"])
  image = _augment_helper(image)  # slice, reshape, resize_bilinear    
  return image, parsed["label"]

def make_dataset():
  dataset = tf.data.TFRecordDataset("/path/to/dataset/train-*.tfrecord")
  dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
  dataset = dataset.map(map_func=parse_fn)
  dataset = dataset.batch(batch_size=FLAGS.batch_size)
  return dataset
```

                .

##  

GPU TPU          CPU    
. `tf.data` API  CPU   ETL       
 .

### 

         .      . ,  
 CPU      . ,     CPU . 
   CPU       .

****      .  `N`     CPU `N+1` 
 .            .

  CPU GPU/TPU    :

![without pipelining](https://www.tensorflow.org/images/datasets_without_pipelining.png)

      :

![with pipelining](https://www.tensorflow.org/images/datasets_with_pipelining.png)

`tf.data` API    `tf.data.Dataset.prefetch`   . 
          . ,       
       (prefetch).         
  .     `tf.data.experimental.AUTOTUNE`  tf.data 
     .

   ,  :

```
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
```

    .

  ""  ""      .

###   

  ,      .   `tf.data` API `tf.data.Dataset.map`
 ,    ( ,  `parse_fn`)     .   
     CPU     .    `map`   
 `num_parallel_calls`  .  ,   `map` 
`num_parallel_calls=2`    .

![parallel map](https://www.tensorflow.org/images/datasets_parallel_map.png)

  `num_parallel_calls`  ,  ( ),   ,  CPU  
   ;    CPU     .  ,   4 
   `num_parallel_calls=4`     . ,
`num_parallel_calls`  CPU          .
`prefetch`   `map`  tf.data     
`tf.data.experimental.AUTOTUNE` .

   ,  :

```
dataset = dataset.map(map_func=parse_fn)
```

 :

```
dataset = dataset.map(map_func=parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
```

###   

                   
 (, GCS HDFS)   .         
           :

*   **  (Time-to-first-byte):**           
        .
*   ** (Read throughput):**             
       .

      /     ( ,
[protobuf](https://developers.google.com/protocol-buffers/)).    
.               
  .

       `tf.data.Dataset.interleave`  (   )
  (interleaving)       .  
`cycle_length`      ,   `num_parallel_calls`   
 . `prefetch` `map`   `interleave` 
`tf.data.experimental.AUTOTUNE` .     tf.data   
.

  `interleave`  `cycle_length=2`  `num_parallel_calls=2`  
 :

![parallel io](https://www.tensorflow.org/images/datasets_parallel_io.png)

     :

```
dataset = tf.data.TFRecordDataset("/path/to/dataset/train-*.tfrecord")
```

 :

```
files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
dataset = files.interleave(
    tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_reads,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
```

##  

`tf.data` API           .   
,      .

###  

`map`              .   
     .  `map`          
.      (,     ) `batch`  `map` 
__   .

###  

`tf.data.Dataset.cache`         .  `map`  
             `map`    . 
                  
.

###  (Interleave) /  / (Shuffle)

`interleave`, `prefetch`, `shuffle`       .   
`map`               .    
    ( ,    (fusing) )      
.

###  

`tf.data.Dataset.repeat`    ( ) ;    
__(epoch) . `tf.data.Dataset.shuffle`     .

`shuffle`   `repeat`     . ,       
  . , `shuffle`  `repeat`    `shuffle`     
       .  , (`shuffle`  `repeat`)   
(`repeat`  `shuffle`)    .

##    

            :

*   `prefetch`      .      `prefetch`
     CPU        .    
    `tf.data.experimental.AUTOTUNE`   tf.data  .
*   `num_parallel_calls`   `map`   .     
    `tf.data.experimental.AUTOTUNE`  tf.data   .
*       (deserialization)      (
    )   `interleave`    .
*   `map`              .
*        , `cache`        ,  
      , ,       .
*       `interleave`, `prefetch`,  `shuffle` () 
        .
*   `repeat`  __ `shuffle`    .
