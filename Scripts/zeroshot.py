# %%
from transformers import pipeline
from datasets import load_dataset, Dataset
from huggingface_hub import login
import evaluate
import numpy as np

login()

checkpoint = "openai/clip-vit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")

# %%
drawings = load_dataset("imagefolder", data_dir="Drawings/")
ds = drawings['train'].to_pandas()

# %%
ds['image'] = ds['image'].apply(lambda x: x.get('path'))

# %%
# Reference: https://huggingface.co/docs/transformers/en/tasks/zero_shot_image_classification
# Load evaluator
accuracy = evaluate.load("accuracy")

def compute_metrics(evals):
    predictions, labels = evals
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# %%
from PIL import Image

preds = []
labels = []

for index, row in ds.iterrows():
    image = Image.open(row['image'])
    pred = detector(image, candidate_labels=[0,1])
    prediction = pred[0]['label']
    preds.append(prediction)
    labels.append(row['label'])

# %%
print(accuracy.compute(predictions=preds, references=labels))


