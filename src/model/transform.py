import torchvision.transforms as transforms

IMG_SIZE = 224

transform_ = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform(img):
    return transform_(img)