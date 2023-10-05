# Kaggle The Simpsons Competition

This is my solution for the [Simpsons Challenge](https://www.kaggle.com/competitions/simpsons-challenge-gft) on Kaggle.  

The solution was implemented in Jupyter Notebooks with Python.

I did not add the train and test data to the repository, but you can get them from Kaggle and add them to the directory data/train and data/test.

I added all the trained models (except the Vision Transformer) to the [trained_models directory](trained_models).

## Results
I tried different neural networks, both "from scratch" and pre-trained ones. For training, I used one GPU of type NVIDIA GeForce RTX 2080 SUPER.  
I used the following notebooks for training and testing:  
* Own networks and RESNET 18: [TRAIN](the-simpsons-train.ipynb), [TEST](the-simpsons-test.ipynb)
* Vision Transformer from Hugging Face: [TRAIN](the-simpsons-train-vit.ipynb), [TEST](the-simpsons-test-vit.ipynb)


Here are the results:  

| Neural Network                                               | Network Architecture                                                                                                                                                                                    | Training Time | Comment                                                                                                                                                                    | Public Score on Kaggle |
|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| [simpsons_neural_network_1.py](simpsons_neural_network_1.py) | 3 convolutional layers including max pooling<br/>2 fully connected layers<br/>dropout layers                                                                                                            | 1h 20m 4s     | [wandb run](https://wandb.ai/hamm-daniel/kaggle-simpsons/runs/89fka4sl/overview?workspace=user-hamm-daniel) <br/>I took the model after 30 epochs although the validation score got worse after ~10 epochs | 0.57738                |
| [simpsons_neural_network_1.py](simpsons_neural_network_1.py) | 3 convolutional layers including max pooling<br/>2 fully connected layers<br/>dropout layers                                                                                                            | 1h 21m 56s    | [wandb run](https://wandb.ai/hamm-daniel/kaggle-simpsons/runs/1c2ehxl7/overview?workspace=user-hamm-daniel) <br/>I took the model with the best validation score <br/>I used `ReduceLROnPlateau` to adjust the learning rate based on the validation score | 0.52759                |
| [simpsons_neural_network_1.py](simpsons_neural_network_1.py) | 3 convolutional layers including max pooling<br/>2 fully connected layers<br/>dropout layers                                                                                                            | 1h 14m 28s    | [wandb run](https://wandb.ai/hamm-daniel/kaggle-simpsons/runs/cxjx9df9/overview?workspace=user-hamm-daniel) Same as last run, but I did not pre-process the training images by randomly rotating and horizontal flipping them | 0.61911                |
| [simpsons_neural_network_2.py](simpsons_neural_network_2.py) | 4 convolutional layers including max pooling<br/>4 batch normalization layers<br/>3 fully connected layers<br/>dropout layers                                                                           | 1h 52m 10s    | [wandb run](https://wandb.ai/hamm-daniel/kaggle-simpsons/runs/mz69lif4/overview?workspace=user-hamm-daniel) <br/>Used a better network architecture                        | 0.71063                |
| [resnet_pretrained.py](resnet_pretrained.py)                 | [Pretrained RESNET 18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)<br/>Replaced the last fully connected layer to the specific number of output classes          | 2h 36m 0s     | [wandb run](https://wandb.ai/hamm-daniel/kaggle-simpsons/runs/1g04rrjo/overview?workspace=user-hamm-daniel) <br/>Used the pre-trained RESNET 18 model                      | 0.81965                |
| [huggingface_pretrained.py](huggingface_pretrained.py)       | [Pretrained Vision Transformer](https://huggingface.co/google/vit-base-patch16-224-in21k)<br/>Vision Transformer (ViT) model pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224. | 1h 33m 44s    | [wandb run](https://wandb.ai/hamm-daniel/kaggle-simpsons/runs/miig6ldy/overview?workspace=user-hamm-daniel) <br/>Used the pre-trained Vision Transformer from Hugging Face | 0.8183                 |