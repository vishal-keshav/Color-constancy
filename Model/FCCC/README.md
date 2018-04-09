# Fully convolutional color constancy with confidence-weighted pooling

This is an **un-official** implementation of network proposed by [This paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Hu_FC4_Fully_Convolutional_CVPR_2017_paper.pdf)
Alternatively, paper can be found in this repostory under Paper section.
Implementation uses tensorflow to construct network graph. Benchmark result will be evaluated and shared here soon!
Each hyperparameters has been used as described in paper. However, training follows without data augmentation.

### Angular loss over first training epoch
![angular_loss][angular_loss]

### Network structure
![network_struct][network_struct]

[angular_loss]: angular_loss.png
[network_struct]: graph.png
