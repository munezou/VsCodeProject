# 

Note:     .           

[  ](https://github.com/tensorflow/docs/blob/master/site/en/guide/variable.md)
    .     
[tensorflow/docs](https://github.com/tensorflow/docs)     
.    
[docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
  .

 ****          .

 `tf.Variable`  .  `tf.Variable`   ,   
   .      . `tf.keras`      
  `tf.Variable` .    `tf.Variable` , ,  
.

##  

     .

``` python
my_variable = tf.Variable(tf.zeros([1., 2., 3.]))
```

   `[1, 2, 3]`  0  3   .    `dtype`
`tf.float32`  . dtype     .

`tf.device` (scope) ,    ;  ,  dtype  " "
 .(GPU     GPU  ) ,   `v` 
   GPU :

``` python
with tf.device("/device:GPU:1"):
  v = tf.Variable(tf.zeros([10, 10]))
```

 `tf.distribute` API         .

##  

  `tf.Variable`     `tf.Tensor`  :

``` python
v = tf.Variable(0.0)
w = v + 1  # w v   tf.Tensor .
           #    ,  
           # tf.Tensor   .
```

   `assign`, `assign_add`  `tf.Variable`   (friends)
. ,      :

``` python
v = tf.Variable(0.0)
v.assign_add(1)
```

  (optimizer)           .
`tf.keras.optimizers.Optimizer`     .

, `read_value`       :

```python
v = tf.Variable(0.0)
v.assign_add(1)
v.read_value()  # 1.0
```

`tf.Variable`    (scope)   .

###  

     . , , ,          
   .

              `tf.Module` 
. `tf.Module`  `variables`           (
)   `trainable_variables`   .

```python
class MyModuleOne(tf.Module):
  def __init__(self):
    self.v0 = tf.Variable(1.0)
    self.vs = [tf.Variable(x) for x in range(10)]
    
class MyOtherModule(tf.Module):
  def __init__(self):
    self.m = MyModuleOne()
    self.v = tf.Variable(10.0)
    
m = MyOtherModule()
len(m.variables)  # 12; 11 m.m   m.v

```

   `tf.keras.Layer`      .      
  `model.fit`    API    . `tf.keras.Layer`  
`tf.Module`   .
