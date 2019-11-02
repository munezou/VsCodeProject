#   

    .            
[  ](https://www.tensorflow.org/?hl=en)     .    
 [tensorflow/docs](https://github.com/tensorflow/docs)     
.    
[docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
  .

: [site/en](https://github.com/tensorflow/docs/tree/master/site/en)
  [ 2](https://www.tensorflow.org)     .
2.0    [site/ko/r1](https://github.com/tensorflow/docs/tree/master/site/ko/r1)   TF 1.x      . 
[](https://groups.google.com/a/tensorflow.org/d/msg/docs/vO0gQnEXcSM/YK_ybv7tBQAJ)
.  install   yaml , index.md   .      [](https://groups.google.com/a/tensorflow.org/forum/#!msg/docs-zh-cn/mhLp-egzNyE/EhGSeIBqAQAJ) .

: [Swift for TensorFlow](https://www.tensorflow.org/swift)(S4TF)  
 
[site/ko/swift](https://github.com/tensorflow/docs/tree/master/site/ko/swift)
.    tensorflow/swift 
[docs/site](https://github.com/tensorflow/swift/tree/master/docs/site)
. S4TF      .

# Community translations

Our TensorFlow community has translated these documents. Because community
translations are *best-effort*, there is no guarantee that this is an accurate
and up-to-date reflection of the
[official English documentation](https://www.tensorflow.org/?hl=en) and [Tensorflow Docs-Ko Translation](http://bit.ly/tf-docs-translation-status).
If you have suggestions to improve this translation, please send a pull request
to the [tensorflow/docs](https://github.com/tensorflow/docs) GitHub repository.
To volunteer to write or review community translations, contact the
[docs@tensorflow.org list](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs).

Note: Please focus translation efforts on
[TensorFlow 2](https://www.tensorflow.org) in the
[site/en](https://github.com/tensorflow/docs/tree/master/site/en)
directory. TF 1.x community docs in [site/en/r1](https://github.com/tensorflow/docs/tree/master/site/en/r1) will no longer be updated as we prepare for the
2.0 release. See
[the announcement](https://groups.google.com/a/tensorflow.org/d/msg/docs/vO0gQnEXcSM/YK_ybv7tBQAJ).
Also, please do not translate the `install/` section, any `*.yaml` files, or `index.md` files.
See [the announcement](https://groups.google.com/a/tensorflow.org/forum/#!msg/docs-zh-cn/mhLp-egzNyE/EhGSeIBqAQAJ).

Note: The
[site/ko/swift](https://github.com/tensorflow/docs/tree/master/site/ko/swift)
directory is the home for
[Swift for TensorFlow](https://www.tensorflow.org/swift)(S4TF) translations.
Original files are in the
[docs/site](https://github.com/tensorflow/swift/tree/master/docs/site) directory
of the tensorflow/swift repository. S4TF notebooks must have the outputs saved.

#   

    .
    [](https://github.com/tensorflow/docs/tree/master/site/ko)
   .
  ''     .
     .

  [   ](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
  [Tensorflow Docs-Ko Translation](http://bit.ly/tf-docs-translation-status)       .
      en    ko      .
site/ko/r1   1.x   .
site/ko   2.x   .

(markdown)   .  (cell)  .
         .
    .
              .
       (Colab)    .
     .

     
[   ](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
    .

!

# For new contributors

Thanks for joining the translation effort.
Please read the existing
[KO documents](https://github.com/tensorflow/docs/tree/master/site/ko)
before starting your translation.
You should use '' style and not use the honorific or rude words.
Follow the style of existing documents, as possible as you can.

After your translation is complete, notify the
[Korean TensorFlow Documentation Contributors](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
mailing list to coordinate a review.

Copy a file in `en` folder to same location under `ko` folder if anybody doesn't work on the file,
and get it start.
`site/ko/r1` are for TensorFlow 1.x.
`site/ko` are for TensorFlow 2.x.

You should translate markdown and comments. You should not run code cells.
Whole file structure can be changed even if you modify only a chunk in the notebook.
It is hard to review such a file in GitHub.
You should use a text editor when you edit a few words of existing notebook.
You should test the notebook in your repository with Colab after you finish the translation.
You can request for review if there is no error.

If you have any question about translation, feel free to contact
[Korean TensorFlow Documentation Contributors](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
mailing list.

Thanks!

# Korean translation guide

Some technical words in English do not have a natural translation. Do *not*
translate the following words:

*   mini-batch
*   batch
*   label
*   class
*   helper
*   hyperparameter
*   optimizer
*   one-hot encoding
*   epoch
*   callback
*   sequence
*   dictionary (in python)
*   embedding
*   padding
*   unit
*   node
*   nueron
*   target
*   checkpoint
*   compile
*   dropout
*   penalty

If you have any suggestion about translation guide, feel free to contact
[Korean TensorFlow Documentation Translation Glossary](http://bit.ly/tf-docs-translation-glossary)
spreadsheet.
