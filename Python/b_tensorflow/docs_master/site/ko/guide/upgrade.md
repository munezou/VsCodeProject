#  2.0  

Note:     .           

[  ](https://github.com/tensorflow/docs/blob/master/site/en/guide/upgrade.md)
    .     
[tensorflow/docs](https://github.com/tensorflow/docs)     
.    
[docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
  .

 2.0  , (symbol)  ,      API  . 
       .    TF 2.0    
   API       `tf_upgrade_v2`  .

`tf_upgrade_v2`  TF 2.0 `pip install`   .    1.x
   2.0    .

            .

##  

 API          .   2.0  ,
`compat.v1`     .   `tf.foo`  TF 1.x  
`tf.compat.v1.foo` .          `tf.compat.v1.*`
(namespace)  `tf.*`    API   .

 2.x   ( , `tf.flags` `tf.contrib`) ,  
`compat.v1`      .      ( ,
`absl.flags`) [tensorflow/addons](http://www.github.com/tensorflow/addons) 
    .

##  

   1.x  2.x ,   :

### pip   

, `pip install` `tensorflow`  `tensorflow-gpu`  .

Note: `tf_upgrade_v2`  1.13     . (nightly TF 2.0 
)

      :

```sh
tf_upgrade_v2 --infile tensorfoo.py --outfile tensorfoo-upgraded.py
```

        .     :

```
# .py      outtree 
tf_upgrade_v2 --intree coolcode --outtree coolcode-upgraded

# .py  
tf_upgrade_v2 --intree coolcode --outtree coolcode-upgraded --copyotherfiles False
```

##  

    , :

```
'tensorflow/tools/compatibility/testdata/test_file_v1_12.py' Line 65
--------------------------------------------------------------------------------

Added keyword 'input' to reordered function 'tf.argmax'
Renamed keyword argument from 'dimension' to 'axis'

    Old:         tf.argmax([[1, 3, 2]], dimension=0))
                                        ~~~~~~~~~~
    New:         tf.argmax(input=[[1, 3, 2]], axis=0))

```

   `report.txt`         . `tf_upgrade_v2` 
       TF 1.x    .
## 

-           . , `tf.argmax` 
    `tf.batch_to_space`            
      .

-    `tensorflow` `import tensorflow as tf` (import)  .

-        .         
    .

-   [tf2up.ml](http://tf2up.ml)           
     .

       
[GitHub](https://github.com/tensorflow/tensorflow/issues).   2.0 
,   !
[TF 2.0 Testing community](https://groups.google.com/a/tensorflow.org/forum/#!forum/testing)
 .   [testing@tensorflow.org](mailto:testing@tensorflow.org)  
.
