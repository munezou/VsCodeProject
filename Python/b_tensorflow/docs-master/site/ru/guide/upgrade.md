# „O„q„~„€„r„y„„„u „{„€„t „t„€ TensorFlow 2.0

TensorFlow 2.0 „r„{„|„„‰„p„u„„ „}„~„€„s„€ „y„x„}„u„~„u„~„y„z API, „„„p„{„y„‡ „{„p„{ „y„x„}„u„~„u„~„y„u „„€„‚„‘„t„{„p
„p„‚„s„…„}„u„~„„„€„r, „„u„‚„u„y„}„u„~„€„r„p„~„y„u „ƒ„y„}„r„€„|„€„r „y „y„x„}„u„~„u„~„y„u „x„~„p„‰„u„~„y„z „„€ „…„}„€„|„‰„p„~„y„
„„p„‚„p„}„u„„„‚„€„r. „B„„„€„|„~„u„~„y„u „„„„y„‡ „}„€„t„y„†„y„{„p„ˆ„y„z „r„‚„…„‰„~„…„ „…„„„€„}„y„„„u„|„Ž„~„€ „y „„€„t„r„u„‚„w„u„~„€
„€„Š„y„q„{„p„}. „D„|„‘ „…„„‚„€„‹„u„~„y„‘ „y„x„}„u„~„u„~„y„z „y „{„p„{ „}„€„w„~„€ „q„€„|„u„u „„|„p„r„~„€„s„€ „„u„‚„u„‡„€„t„p „~„p TF 2.0,
„{„€„}„p„~„t„p TensorFlow „ƒ„€„x„t„p„|„p „…„„„y„|„y„„„… `tf_upgrade_v2`, „„€„}„€„s„p„„‹„…„ „„u„‚„u„z„„„y „€„„
„…„ƒ„„„p„‚„u„r„Š„u„s„€ „{„€„t„p „{ „~„€„r„€„}„… API.

„T„„„y„|„y„„„p `tf_upgrade_v2` „t„€„q„p„r„|„‘„u„„„ƒ„‘ „p„r„„„€„}„p„„„y„‰„u„ƒ„{„y „ƒ `pip install` TF 2.0. „O„~„p
„…„ƒ„{„€„‚„y„„ „„‚„€„ˆ„u„ƒ„ƒ „€„q„~„€„r„|„u„~„y„‘ „x„p „ƒ„‰„u„„ „„‚„u„€„q„‚„p„x„€„r„p„~„y„‘ „ƒ„…„‹„u„ƒ„„„r„…„„‹„y„‡ „ƒ„{„‚„y„„„„€„r
TensorFlow 1.x Python „r TensorFlow 2.0.

„R„{„‚„y„„„ „€„q„~„€„r„|„u„~„y„‘ „p„r„„„€„}„p„„„y„x„y„‚„…„u„„ „}„p„{„ƒ„y„}„…„} „r„€„x„}„€„w„~„€„s„€, „~„€ „r„ƒ„u „u„‹„u „u„ƒ„„„Ž
„ƒ„y„~„„„p„{„ƒ„y„‰„u„ƒ„{„y„u „y „ƒ„„„y„|„y„ƒ„„„y„‰„u„ƒ„{„y„u „y„x„}„u„~„u„~„y„‘, „{„€„„„€„‚„„u „~„u „}„€„s„…„„ „q„„„„Ž „r„„„€„|„~„u„~„
„ƒ„{„‚„y„„„„€„}.

## „M„€„t„…„|„Ž „ƒ„€„r„}„u„ƒ„„„y„}„€„ƒ„„„y

„N„u„{„€„„„€„‚„„u „ƒ„y„}„r„€„|„ API „~„u „}„€„s„…„„ „q„„„„Ž „€„q„~„€„r„|„u„~„ „„‚„€„ƒ„„„€ „ƒ „„€„}„€„‹„Ž„ „x„p„}„u„~„ „ƒ„„„‚„€„{„y.
„X„„„€„q„ „s„p„‚„p„~„„„y„‚„€„r„p„„„Ž „„€„t„t„u„‚„w„{„… „r„p„Š„u„s„€ „{„€„t„p „r TensorFlow 2.0, „ƒ„{„‚„y„„„ „€„q„~„€„r„|„u„~„y„‘
„r„{„|„„‰„p„u„„ „r „ƒ„u„q„‘ „}„€„t„…„|„Ž `compat.v1`. „^„„„€„„ „}„€„t„…„|„Ž „x„p„}„u„~„‘„u„„ TF 1.x „ƒ„y„}„r„€„|„
„~„p„„€„t„€„q„y„u `tf.foo` „y„‡ „„{„r„y„r„p„|„u„~„„„p„}„y `tf.compat.v1.foo`. „V„€„„„‘ „}„€„t„…„|„Ž
„ƒ„€„r„}„u„ƒ„„„y„}„€„ƒ„„„y „‡„€„‚„€„Š, „}„ „‚„u„{„€„}„u„~„t„…„u„} „r„p„} „r„‚„…„‰„~„…„ „r„„‰„y„„„p„„„Ž „x„p„}„u„~„ „y „„u„‚„u„~„u„ƒ„„„y „y„‡
„~„p „~„€„r„„u API „r „„‚„€„ƒ„„„‚„p„~„ƒ„„„r„u „y„}„u„~ `tf. *` „r„}„u„ƒ„„„€ „„‚„€„ƒ„„„‚„p„~„ƒ„„„r„p „y„}„u„~ `tf.compat.v1.
*`.

„I„x-„x„p „…„ƒ„„„p„‚„u„r„p„~„y„‘ „}„€„t„…„|„u„z TensorFlow 2.x („~„p„„‚„y„}„u„‚, `tf.flags` „y`tf.contrib`)
„~„u„{„€„„„€„‚„„u „y„x„}„u„~„u„~„y„‘ „~„u „}„€„s„…„„ „q„„„„Ž „€„„„‚„p„q„€„„„p„~„ „„…„„„u„} „„u„‚„u„{„|„„‰„u„~„y„‘ „~„p `compat.v1`.
„O„q„~„€„r„|„u„~„y„u „„„„€„s„€ „{„€„t„p „}„€„w„u„„ „„€„„„‚„u„q„€„r„p„„„Ž „y„ƒ„„€„|„Ž„x„€„r„p„~„y„‘ „t„€„„€„|„~„y„„„u„|„Ž„~„€„z „q„y„q„|„y„€„„„u„{„y
(„~„p„„‚„y„}„u„‚, áabsl.flagsâ) „y„|„y „„u„‚„u„{„|„„‰„u„~„y„‘ „~„p „„p„{„u„„ „r
[tenorflow / addons](http://www.github.com/tensorflow/addons).

## „R„{„‚„y„„„ „€„q„~„€„r„|„u„~„y„‘

„X„„„€„q„ „{„€„~„r„u„‚„„„y„‚„€„r„p„„„Ž „r„p„Š „{„€„t „y„x TensorFlow 1.x „r TensorFlow 2.x, „ƒ„|„u„t„…„z„„„u
„ƒ„|„u„t„…„„‹„y„} „y„~„ƒ„„„‚„…„{„ˆ„y„‘„}:

### „H„p„„…„ƒ„„„y„„„u „ƒ„{„‚„y„„„ „y„x „„p„{„u„„„p pip

„R„„u„‚„r„p „ƒ `pip install` „…„ƒ„„„p„~„€„r„y„„„u „„p„{„u„„ `tensorflow==2.0.0-beta0` „y„|„y
`tensorflow-gpu==2.0.0-beta0`.

„P„‚„y„}„u„‰„p„~„y„u: `tf_upgrade_v2` „…„ƒ„„„p„~„p„r„|„y„r„p„u„„„ƒ„‘ „p„r„„„€„}„p„„„y„‰„u„ƒ„{„y „t„|„‘ TensorFlow 1.13 „y
„r„„Š„u („r„{„|„„‰„p„‘ „~„€„‰„~„„u „ƒ„q„€„‚„{„y TF 2.0).

„R„{„‚„y„„„ „€„q„~„€„r„|„u„~„y„‘ „}„€„w„u„„ „q„„„„Ž „x„p„„…„‹„u„~ „~„p „€„t„~„€„} „†„p„z„|„u Python:

```sh
tf_upgrade_v2 --infile tensorfoo.py --outfile tensorfoo-upgraded.py
```

„R„{„‚„y„„„ „r„„r„u„t„u„„ „€„Š„y„q„{„y „u„ƒ„|„y „€„~ „~„u „ƒ„}„€„w„u„„ „~„p„z„„„y „y„ƒ„„‚„p„r„|„u„~„y„u „t„|„‘ „{„€„t„p. „B„ „„„p„{„w„u
„}„€„w„u„„„u „x„p„„…„ƒ„„„y„„„Ž „u„s„€ „~„p „t„u„‚„u„r„u „{„p„„„p„|„€„s„€„r:

```
# „€„q„~„€„r„y„„„u „†„p„z„|„ .py „y „ƒ„{„€„„y„‚„…„z„„„u „r„ƒ„u „€„ƒ„„„p„|„Ž„~„„u „†„p„z„|„ „r outtree
tf_upgrade_v2 --intree coolcode --outtree coolcode-upgraded

# „€„q„~„€„r„|„u„~„y„u „„„€„|„Ž„{„€ .py „†„p„z„|„€„r
tf_upgrade_v2 --intree coolcode --outtree coolcode-upgraded --copyotherfiles False
```

## „D„u„„„p„|„Ž„~„„z „€„„„‰„u„„

„R„{„‚„y„„„ „„„p„{„w„u „ƒ„€„€„q„‹„p„u„„ „ƒ„„y„ƒ„€„{ „t„u„„„p„|„Ž„~„„‡ „y„x„}„u„~„u„~„y„z, „~„p„„‚„y„}„u„‚:

```
'tensorflow/tools/compatibility/testdata/test_file_v1_12.py' Line 65
--------------------------------------------------------------------------------

Added keyword 'input' to reordered function 'tf.argmax'
Renamed keyword argument from 'dimension' to 'axis'

    Old:         tf.argmax([[1, 3, 2]], dimension=0))
                                        ~~~~~~~~~~
    New:         tf.argmax(input=[[1, 3, 2]], axis=0))

```

„B„ƒ„‘ „„„„p „y„~„†„€„‚„}„p„ˆ„y„‘ „t„€„q„p„r„|„‘„u„„„ƒ„‘ „r „†„p„z„|`report.txt`, „{„€„„„€„‚„„z „q„…„t„u„„ „„{„ƒ„„€„‚„„„y„‚„€„r„p„~ „r
„r„p„Š„… „„„u„{„…„‹„…„ „„p„„{„…. „P„€„ƒ„|„u „r„„„€„|„~„u„~„y„‘ `tf_upgrade_v2` „y „„{„ƒ„„€„‚„„„p „r„p„Š„u„s„€
„€„q„~„€„r„|„u„~„~„€„s„€ „ƒ„{„‚„y„„„„p, „r„ „}„€„w„u„„„u „x„p„„…„ƒ„„„y„„„Ž „}„€„t„u„|„Ž „y „…„q„u„t„y„„„Ž„ƒ„‘, „‰„„„€ „‚„u„x„…„|„Ž„„„p„„
„p„~„p„|„€„s„y„‰„u„~ TF 1.x.

## „P„‚„u„t„€„ƒ„„„u„‚„u„w„u„~„y„‘

-   „N„u „€„q„~„€„r„|„‘„z„„„u „‰„p„ƒ„„„y „r„p„Š„u„s„€ „{„€„t„p „r„‚„…„‰„~„…„ „t„€ „x„p„„…„ƒ„{„p „„„„€„s„€ „ƒ„{„‚„y„„„„p. „B
    „‰„p„ƒ„„„~„€„ƒ„„„y, „†„…„~„{„ˆ„y„y, „„€„}„u„~„‘„r„Š„y„u „„€„‚„‘„t„€„{ „p„‚„s„…„}„u„~„„„€„r, „„„p„{„y„u „{„p„{ `tf.argmax` „y„|„y
    `tf.batch_to_space` „r„„~„…„t„‘„„ „ƒ„{„‚„y„„„ „~„u„„‚„p„r„y„|„Ž„~„€ „t„€„q„p„r„y„„„Ž „y„}„u„~„p „p„‚„s„…„}„u„~„„„€„r,
    „‰„„„€ „„‚„y„r„u„t„u„„ „{ „€„Š„y„q„{„p„} „r „ƒ„…„‹„u„ƒ„„„r„…„„‹„u„} „{„€„t„u.

-   „R„{„‚„y„„„ „„‚„u„t„„€„|„p„s„p„u„„ „‰„„„€ `tensorflow` „y„}„„€„‚„„„y„‚„€„r„p„~ „ƒ „y„ƒ„„€„|„Ž„x„€„r„p„~„y„u„} `import
    tensorflow as tf`.

-   „^„„„€„„ „ƒ„{„‚„y„„„ „~„u „}„u„~„‘„u„„ „p„‚„s„…„}„u„~„„„. „B„}„u„ƒ„„„€ „„„„€„s„€ „ƒ„{„‚„y„„„ „t„€„q„p„r„|„‘„u„„ „{„|„„‰„y
    „p„‚„s„…„}„u„~„„„€„r „{ „†„…„~„{„ˆ„y„‘„}, „… „{„€„„„€„‚„„‡ „y„x„}„u„~„y„„„ƒ„‘ „„€„‚„‘„t„€„{ „p„‚„s„…„}„u„~„„„€„r.

-   „P„‚„€„r„u„‚„Ž„„„u [tf2up.ml](http://tf2up.ml) „t„|„‘ „…„t„€„q„~„€„s„€ „y„~„ƒ„„„‚„…„}„u„~„„„p „€„q„~„€„r„|„u„~„y„‘
    „†„p„z„|„€„r „r „†„€„‚„}„p„„„p„‡ Jupyter Notebook „y Python „r „‚„u„„€„x„y„„„€„‚„y„y GitHub.

„X„„„€„q„ „ƒ„€„€„q„‹„y„„„Ž „€„q „€„Š„y„q„{„p„‡ „r „ƒ„{„‚„y„„„„u „€„q„~„€„r„|„u„~„y„‘ „y„|„y „€„„„„‚„p„r„y„„„Ž „x„p„„‚„€„ƒ „~„p
„t„€„q„p„r„|„u„~„y„u „~„€„r„„‡ „†„…„~„{„ˆ„y„z, „€„„„„‚„p„r„Ž„„„u „ƒ„€„€„q„‹„u„~„y„u „€„q „€„Š„y„q„{„u „~„p
[GitHub](https://github.com/tensorflow/tensorflow/issues). „I „u„ƒ„|„y „r„ „„„u„ƒ„„„y„‚„…„u„„„u
TensorFlow 2.0, „}„ „‡„€„„„y„} „…„ƒ„|„„Š„p„„„Ž „€„q „„„„€„}! „P„‚„y„ƒ„€„u„t„y„~„‘„z„„„u„ƒ„Ž „{ „ƒ„€„€„q„‹„u„ƒ„„„r„…
[TF 2.0 Testing](https://groups.google.com/a/tensorflow.org/forum/#!forum/testing)
„y „€„„„„‚„p„r„|„‘„z„„„u „r„€„„‚„€„ƒ„ „y „€„q„ƒ„…„w„t„u„~„y„‘ „~„p „ƒ„p„z„„
[testing@tensorflow.org](mailto:testing@tensorflow.org).
