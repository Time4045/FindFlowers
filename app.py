from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms, models
from PIL import Image
import pickle
import numpy as np
import random
import time
import json
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Установка random_seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

app = FastAPI()

output_dir = 'test_images'
data = []
for class_name in os.listdir(output_dir):
    class_dir = os.path.join(output_dir, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            data.append((os.path.join(class_dir, img_name), class_name))

test_df = pd.DataFrame(data, columns=["image_path", "label"])

# Загрузка модели
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 5)  # Пример классификации на 5 классов
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Преобразования для входных данных
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Функция для получения эмбеддинга
def get_embedding(image):
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image).squeeze().numpy()
    return embedding

# Получение эмбеддингов
test_embeddings = {}
for _, row in test_df.iterrows():
    image_path = row["image_path"]
    embedding = get_embedding(Image.open(image_path))
    test_embeddings[image_path] = embedding
with open("test_embeddings.pkl", "wb") as f:
    pickle.dump(test_embeddings, f)

# Функция для поиска топ-5 похожих изображений
def find_top_similar_images(input_embedding, test_embeddings, top_k=5):
    similarities = []
    for image_path, embedding in test_embeddings.items():
        similarity = cosine_similarity([input_embedding], [embedding])[0][0]
        similarities.append((image_path, similarity))

    # Сортируем по убыванию сходства
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Возвращаем топ-k результатов
    return similarities[:top_k]

# API-эндпоинт
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Проверка файла
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(status_code=400, content={"error": "Unsupported file format"})

    # Загрузка изображения
    image = Image.open(file.file).convert("RGB")
    input_embedding = get_embedding(image)

    # Поиск топ-5 похожих изображений
    top_similar_images = find_top_similar_images(input_embedding, test_embeddings)

    # Формирование ответа
    result = {image_path: float(similarity) for image_path, similarity in top_similar_images}
    # Сохранение в JSON файл
    try:
        timestamp = int(time.time())
        filename = f"prediction_results_{timestamp}.json"

        # Сохраняем результат в файл
        with open(filename, "w") as outfile:
            json.dump({"results": result}, outfile, indent=4)

        return {
            "status": "success",
            "message": f"Predictions saved to {filename}",
            "results": result  # Оставляем возврат данных для API
        }
    except Exception as e:
        return {"error": f"Failed to save results: {str(e)}"}