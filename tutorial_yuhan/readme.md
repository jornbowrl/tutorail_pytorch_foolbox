# The tutorial of the foolbox and the basic training of pytorch. 

## Install the preparation 
```
#### recommend starting from py37
# conda create -n py37 python=3.7 
# conda install pytorch  torchvision 


pip install -r ./req.txt
```

## Train an image classifier
* Load and normalizing the CIFAR10 training and test datasets using torchvision
* Define a Convolutional Neural Network
* Define a loss function
* Train the network on the training data
* Test the network on the test data


```
#cd tutorial_yuhan
CUDA_VISIBLE_DEVICES=0 python cifar10_tutorial.py

# please check the dataset's root, like "~/.torch/" or "/data1/zyh/copycat/Framework/data"
```

## Try the foolbox to generate the adversiral images 

```
#cd tutorial_yuhan
python foolbox1.py

ls image/*

# please check the dataset's root, like "~/.torch/" or "/data1/zyh/copycat/Framework/data"

```
 

## Related Projects
* [Foolbox](https://github.com/bethgelab/foolbox/blob/master/examples/single_attack_pytorch_resnet18.py)

* [Cifar tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) 




