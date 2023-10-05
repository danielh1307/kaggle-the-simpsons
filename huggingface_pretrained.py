from transformers import ViTForImageClassification, ViTFeatureExtractor


# Vision Transformer (ViT) model pre-trained on ImageNet-21k
# (14 million images, 21,843 classes) at resolution 224x224.
# see https://huggingface.co/google/vit-base-patch16-224-in21k
def get_vision_transformer():
    return ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=29)
