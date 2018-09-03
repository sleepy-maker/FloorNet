# FloorNet: A Unified Framework for Floorplan Reconstruction from 3D Scans
By Chen Liu<sup>\*</sup>, Jiaye Wu<sup>\*</sup>, and Yasutaka Furukawa (<sup>\*</sup> indicates equal contribution)

## Introduction

This paper proposes FloorNet, a novel neural network, to turn RGBD videos of indoor spaces into vector-graphics floorplans. FloorNet consists of three branches, PointNet branch, Floorplan branch, and Image branch. For more details, please refer to our Arxiv [paper](https://arxiv.org/abs/1804.00090) or visit our [project website](http://art-programmer.github.io/floornet.html). This is a follow-up work of our floorplan transformation project which you can find [here](https://github.com/art-programmer/FloorplanTransformation).

## Dependencies
Python 2.7, TensorFlow (>= 1.3), numpy, opencv 3.

## tfrecords
https://github.com/warmspringwinds/tensorflow_notes/blob/master/tfrecords_guide.ipynb      

## Data
We collect 155 scans of residential units and annotated corresponding floorplan information. Among 155 scans, 135 are used for training and 20 are for testing. We convert both training data and testing data to tfrecords files which can be downloaded [here](https://mega.nz/#F!5yQy0b5T!ykkR4dqwGO9J5EwnKT_GBw). Please put the downloaded files under folder *data/*.

## Annotator
For reference, a similar (but not the same) annotator written in Python is [here](https://github.com/art-programmer/FloorplanAnnotator). You need to make some changes to annotate your own data.

## Training
To train the network from scratch, please run:
```bash
python train.py --restore=0
```

## Evaluation
To evaluate the performance of our trained model, please run:
```bash
python train.py --task=evaluate
```


## File Structure Summary

- QP.py

    For solving the IP in the final step? Not sure
    
- train.py

    The model definition of floornet(in build_graph()), and the training/testing/visualization logics.  
    predict hasn't defined yet.
    
- evaluate.py

    Evaluate the final results (produced by Floornet + IP)

- RecordWriterTango.py

    This is for converting the collected Tango data into tfrecords, then during training/testing tfrecords are used to create 
    tensorflow Dataset instances.
    file format .obj/.mtl
    
    RecordWriterTango  
    args  
    numPoints = 50000  
    numInputChannels = 7
    augmentation = ''  
    task = 'val'
    
    
- RecordReader.py

    This is for reading the saved tfrecords when instantiating tensorflow Dataset 

- augmentation_tf.py

    Where is this file used?
    
- floorplan_utils.py

    Is this for drawing the final floorplan output?

## Contact

If you have any questions, please contact me at chenliu@wustl.edu.


### arguments
```
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Planenet')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default='0', type=str)
    # task: [train, test, predict]
    parser.add_argument('--task', dest='task',
                        help='task type: [train, test, predict]',
                        default='train', type=str)
    parser.add_argument('--restore', dest='restore',
                        help='how to restore the model',
                        default=1, type=int)
    parser.add_argument('--batchSize', dest='batchSize',
                        help='batch size',
                        default=4, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name for test/predict',
                        default='5', type=str)
    parser.add_argument('--slice', dest='slice', help='whether or not to use the slice version.',
                        action='store_true')
    parser.add_argument('--numTestingImages', dest='numTestingImages',
                        help='the number of images to test/predict',
                        default=20, type=int)
    parser.add_argument('--fineTuningCheckpoint', dest='fineTuningCheckpoint',
                        help='specify the model for fine-tuning',
                        default='checkpoint/floornet_hybrid4_branch0123_wsf', type=str)
    parser.add_argument('--suffix', dest='suffix',
                        help='add a suffix to keyname to distinguish experiments',
                        default='', type=str)
    parser.add_argument('--l2Weight', dest='l2Weight',
                        help='L2 regulation weight',
                        default=5e-4, type=float)
    parser.add_argument('--LR', dest='LR',
                        help='learning rate',
                        default=3e-5, type=float)
    parser.add_argument('--hybrid', dest='hybrid',
                        help='hybrid training',
                        default='1', type=str)
    parser.add_argument('--branches',
                        help='active branches of the network: 0: PointNet, 1: top-down, 2: bottom-up, 3: PointNet segmentation, 4: Image Features, 5: Image Features with Joint training, 6: Additional Layers Before Pred (0, 01, 012, 0123, 01234, 1, 02*, 013)',
                        default='0123', type=str)
    # parser.add_argument('--batch_norm', help='add batch normalization to network', action='store_true')

    parser.add_argument('--cornerLossType', dest='cornerLossType',
                        help='corner loss type',
                        default='sigmoid', type=str)
    parser.add_argument('--loss',
                        help='loss type needed. [wall corner loss, door corner loss, icon corner loss, icon segmentation, room segmentation]',
                        default='01234', type=str)
    parser.add_argument('--cornerLossWeight', dest='cornerLossWeight',
                        help='corner loss weight',
                        default=10, type=float)
    parser.add_argument('--augmentation', dest='augmentation',
                        help='augmentation (wsfd)',
                        default='wsf', type=str)
    parser.add_argument('--numPoints', dest='numPoints',
                        help='number of points',
                        default=50000, type=int)
    parser.add_argument('--numInputChannels', dest='numInputChannels',
                        help='number of input channels',
                        default=7, type=int)
    parser.add_argument('--sumScale', dest='sumScale',
                        help='avoid segment sum results to be too large',
                        default=10, type=int)
    parser.add_argument('--visualizeReconstruction', dest='visualizeReconstruction',
                        help='whether to visualize flooplan reconstruction or not',
                        default=0, type=int)
    parser.add_argument('--numFinalChannels', dest='numFinalChannels', help='the number of final channels', default=256,
                        type=int)
    parser.add_argument('--numIterations', dest='numIterations', help='the number of iterations', default=10000,
                        type=int)
    parser.add_argument('--startIteration', dest='startIteration', help='the index of iteration to start', default=0,
                        type=int)
    parser.add_argument('--useCache', dest='useCache',
                        help='whether to cache or not',
                        default=1, type=int)
    parser.add_argument('--debug', dest='debug',
                        help='debug index',
                        default=-1, type=int)
    parser.add_argument('--outputLayers', dest='outputLayers',
                        help='output layers',
                        default='two', type=str)
    parser.add_argument('--kernelSize', dest='kernelSize',
                        help='corner kernel size',
                        default=11, type=int)
    parser.add_argument('--iconLossWeight', dest='iconLossWeight',
                        help='icon loss weight',
                        default=1, type=float)
    parser.add_argument('--poolingTypes', dest='poolingTypes',
                        help='pooling types',
                        default='sssmm', type=str)
    parser.add_argument('--visualize', dest='visualize',
                        help='visualize during training',
                        action='store_false')
    parser.add_argument('--iconPositiveWeight', dest='iconPositiveWeight',
                        help='icon positive weight',
                        default=10, type=int)
    parser.add_argument('--prefix', dest='prefix',
                        help='prefix',
                        default='floornet', type=str)
    parser.add_argument('--drawFinal', dest='drawFinal',
                        help='draw final',
                        action='store_false')
    parser.add_argument('--separateIconLoss', dest='separateIconLoss',
                        help='separate loss for icon',
                        action='store_false')
    parser.add_argument('--evaluateImage', dest='evaluateImage',
                        help='evaluate image',
                        action='store_true')
```
