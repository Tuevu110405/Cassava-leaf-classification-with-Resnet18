import torch
from PIL import Image

image_path = ""

image = Image.open(image_path)
image = tesr_val_transforms(image)

print(type(image))

classifier = MyResNet18(5).to(device)
classifier.load_state_dict(torch.load('/content/models/best_model.pth'))
classifier.eval()
result = classifier(image.unsqueeze(0).to(device))
print(result)
pred = torch.argmax(result, dim=1)
labels = {'cbb': 0, 'cbsd': 1, 'cgm': 2, 'cmd': 3, 'healthy': 4}
for key, value in labels.items():
    if value == pred.item():
        print(f'Predicted label: {key}')
        break