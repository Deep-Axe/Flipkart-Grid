import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models


class FruitFreshnessClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FruitFreshnessClassifier, self).__init__()

        self.googlenet = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        self.resnext = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        self.densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        self.googlenet.fc = nn.Identity()
        self.resnext.fc = nn.Identity()
        self.densenet.classifier = nn.Identity()

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(1024 + 2048 + 1920, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        g_features = self.googlenet(x)
        r_features = self.resnext(x)
        d_features = self.densenet(x)

        combined_features = torch.cat((g_features, r_features, d_features), dim=1)
        output = self.fusion(combined_features)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 28 
model = FruitFreshnessClassifier(num_classes)
model.load_state_dict(torch.load('best_model_weights.pth', map_location=device))
model.to(device)
model.eval()

class_names = ['fresh_apple', 'fresh_banana', 'fresh_bitter_gourd', 'fresh_capsicum', 
               'fresh_carrot', 'fresh_cucumber', 'fresh_guava', 'fresh_lime', 
               'fresh_mango', 'fresh_orange', 'fresh_pomegranate', 'fresh_potato', 
               'fresh_strawberry', 'fresh_tomato', 'stale_apple', 'stale_banana', 
               'stale_bitter_gourd', 'stale_capsicum', 'stale_carrot', 'stale_cucumber', 
               'stale_guava', 'stale_lime', 'stale_mango', 'stale_orange', 
               'stale_pomegranate', 'stale_potato', 'stale_strawberry', 'stale_tomato']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(rgb_frame)

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)

    predicted_class_idx = torch.argmax(probabilities, 1).item()
    confidence = probabilities[0][predicted_class_idx].item()
    class_name = class_names[predicted_class_idx]

    label = f"{class_name}: {confidence * 100:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Fruit Freshness Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
