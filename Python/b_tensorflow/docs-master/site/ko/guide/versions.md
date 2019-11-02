#   

Note:     .           

[  ](https://github.com/tensorflow/docs/blob/master/site/en/guide/versions.md)
    .     
[tensorflow/docs](https://github.com/tensorflow/docs)     
.    
[docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
  .

  ( )            
  .

##   2.0

  API   2.0([semver](http://semver.org)) .    
`MAJOR.MINOR.PATCH` .    1.2.3 `MAJOR`  1, `MINOR`  2,
`PATCH`  3 .     :

*   **MAJOR**:      .  (major)      
       . ,           ;
      [  ](#compatibility_of_graphs_and_checkpoints) .

*   **MINOR**:   ,    .  (minor) **      API
        .  API    
    [ ](#What_is_covered) .

*   **PATCH**:    

  1.0.0  0.12.1   **  . ,  1.1.1  1.0.0
  .

##  

  API      .  API  .

*     [](https://www.tensorflow.org/api_docs/python) `tensorflow` 
       ,  

    *    (private symbol): `_`     
    *     `tf.contrib` ,  [](#not_covered) .

    `examples/` `tools/`    `tensorflow`        
      .

      `tensorflow`       ,  API  
    ****.

*    API( `tf.compat` ).          
        .  API    (,     
      )  .

*   [C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h).

*      :

    *   [`attr_value`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)
    *   [`config`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto)
    *   [`event`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto)
    *   [`graph`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)
    *   [`op_def`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)
    *   [`reader_base`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/reader_base.proto)
    *   [`summary`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto)
    *   [`tensor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)
    *   [`tensor_shape`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto)
    *   [`types`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto)

<a name="not_covered"></a>
##  ** 

          .   :

*   ** API**:    ,  API        .
          :

    -   `tf.contrib`     
    -   `experimental`  `Experimental`    (, , , ,
        , ); 
    -            . `experimental`   
           .

*   ** :**  C   API , :

    -   [C++](https://www.tensorflow.org/api_guides/cc/guide.md)
        ([`tensorflow/cc`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc)
           ).
    -   [Java](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary),
    -   [Go](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)
    -   [JavaScript](https://js.tensorflow.org)

*   **  :**        ,   
    `GraphDef`     .   (minor)    . ,
                    
        .

*   **  :**           .    
     ,       .        .
    (          .)

*   ** :**   
    [random ops](https://www.tensorflow.org/api_guides/python/constant_op.md#Random_Tensors)
         .      ,      
     . ,        .     .

*   **   :**           . 
    (wire protocol)     .

*   **:**      ,        .   
                    . , 
       (optimizer)       ,
    (optimizer)  .         . 
        .

*   ** :**     . ,        
      .       . ,        
          .

##   ,  

       (serialization format) .    
 :      `GraphDefs`   .     
      .

          .
[semver](https://semver.org)            
    .

**     .    `N` **     
API**     * `N` .*    `N`      
 `N+1`    . ,             , 
     .

         .

### GraphDef compatibility

Graphs are serialized via the `GraphDef` protocol buffer.  To facilitate
backwards incompatible changes to graphs, each `GraphDef` has a version number
separate from the TensorFlow version.  For example, `GraphDef` version 17
deprecated the `inv` op in favor of `reciprocal`.  The semantics are:

* Each version of TensorFlow supports an interval of `GraphDef` versions. This
  interval will be constant across patch releases, and will only grow across
  minor releases.  Dropping support for a `GraphDef` version will only occur
  for a major release of TensorFlow (and only aligned with the version support
  guaranteed for SavedModels).

* Newly created graphs are assigned the latest `GraphDef` version number.

* If a given version of TensorFlow supports the `GraphDef` version of a graph,
  it will load and evaluate with the same behavior as the TensorFlow version
  used to generate it (except for floating point numerical details and random
  numbers as outlined above), regardless of the major version of TensorFlow.
  In particular, a GraphDef which is compatible with a checkpoint file in one
  version of TensorFlow (such as is the case in a SavedModel) will remain
  compatible with that checkpoint in subsequent versions, as long as the
  GraphDef is supported.

  Note that this applies only to serialized Graphs in GraphDefs (and
  SavedModels): *Code* which reads a checkpoint may not be able to read
  checkpoints generated by the same code running a different version of
  TensorFlow.

* If the `GraphDef` *upper* bound is increased to X in a (minor) release, there
  will be at least six months before the *lower* bound is increased to X.  For
  example (we're using hypothetical version numbers here):

    * TensorFlow 1.2 might support `GraphDef` versions 4 to 7.
    * TensorFlow 1.3 could add `GraphDef` version 8 and support versions 4 to 8.
    * At least six months later, TensorFlow 2.0.0 could drop support for
      versions 4 to 7, leaving version 8 only.

  Note that because major versions of TensorFlow are usually published more than
  6 months apart, the guarantees for supported SavedModels detailed above are
  much stronger than the 6 months guarantee for GraphDefs.

Finally, when support for a `GraphDef` version is dropped, we will attempt to
provide tools for automatically converting graphs to a newer supported
`GraphDef` version.

## Graph and checkpoint compatibility when extending TensorFlow

This section is relevant only when making incompatible changes to the `GraphDef`
format, such as when adding ops, removing ops, or changing the functionality
of existing ops.  The previous section should suffice for most users.

<a id="backward_forward"/>

### Backward and partial forward compatibility

    :

*           ** **.
*     (producer) (consumer)      
    ** **.
*          . ,      .

`GraphDef`      , `GraphDef`      
 . ,  ` (MAJOR)` ( `1.7` `2.0`)     . 
  ( `1.x.1` `1.x.2`) .

            ,     
 .     `GraphDef`     .

###    

      .            
.    
[`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h).
          .

### , ,  

      : * ****:   .  (`producer`) 
    (`min_consumer`) . * ****:   . 
(`consumer`)      (`min_producer`) .

    ``   `min_consumer`,    `bad_consumers` 
 
[`VersionDef versions`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/versions.proto)
  .

,      `producer` `min_consumer`  .  
       `bad_consumers`   .     
   :

*   `consumer` >=  `min_consumer`
*    `producer` >=  `min_producer`
*    `bad_consumers`  `consumer`

         ,
[`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h)
 `min_consumer` `min_producer`(    )  `producer`
`consumer`        . ,

*   `GraphDef`  `TF_GRAPH_DEF_VERSION`, `TF_GRAPH_DEF_VERSION_MIN_CONSUMER`,
    `TF_GRAPH_DEF_VERSION_MIN_PRODUCER`.
*     `TF_CHECKPOINT_VERSION`, `TF_CHECKPOINT_VERSION_MIN_CONSUMER`,
    `TF_CHECKPOINT_VERSION_MIN_PRODUCER`.

###      

          :

1.    , `SavedModelBuilder`
    `tf.saved_model.SavedModelBuilder.add_meta_graph_and_variables`
    `tf.saved_model.SavedModelBuilder.add_meta_graph` 
    `tf.estimator.Estimator.export_saved_model`    
    `strip_default_attrs` `True` .
2.           .
3.      (,     )       
         .

### GraphDef 

  `GraphDef`         .

####   

`GraphDef`        .      
                .

####         

1.      `GraphDef`  .
2.           ,    .
3.       .         `min_consumer`
     .

####    

1.          (  ) .
2.  `GraphDef`    GraphDef         
    .     `GraphDefs`   .  
    [`REGISTER_OP(...).Deprecated(deprecated_at_version, message)`](https://github.com/tensorflow/tensorflow/blob/b289bc7a50fc0254970c60aaeba01c33de61a728/tensorflow/core/ops/array_ops.cc#L1009)
    .
3.       .
4.  (2) GraphDef  `min_producer`    .

####   

1.  `SomethingV2`                
    .       
    [compat.py](https://www.tensorflow.org/code/tensorflow/python/compat/compat.py)
      .
2.    (     ).
3.        `min_consumer`    `SomethingV2`  
    .          .
4.  `SomethingV2`   .

####     

1.  `GraphDef`       GraphDef `bad_consumers` . 
         GraphDef `bad_consumers` .
2.      ,   .
