import librosa
from sklearn.metrics import classification_report
from datasets import load_dataset, load_metric
import torchaudio
import torch
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from src.models import Wav2Vec2ForSpeechClassification
from glob import glob

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    #speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, feature_extractor.sampling_rate)
    batch["speech"] = speech_array
    return batch


def predict(batch):
    features = feature_extractor(batch["speech"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits 

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

test_dataset = load_dataset("csv", data_files={"test": "dataset/test.csv"}, delimiter="\t")["test"] # loading dataset generated by preprocess_data.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Device: {device}")

model_name_or_path = glob("ckps/wav2vec2-base-100k-classifier/checkpoint*")[0] # specify path of trained model
print('testing on models ',model_name_or_path)
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

max_samples = 1000
idxes = []
cnt = 0
for i in range(len(test_dataset)):
    if cnt < max_samples//2:
        if test_dataset[i]['label'] == 0:
            idxes.append(i)
            cnt += 1
    else:
        break
cnt = 0
for i in range(len(test_dataset)):
    if cnt < max_samples//2:
        if test_dataset[i]['label'] == 1:
            idxes.append(i)
            cnt += 1
    else:
        break
test_dataset = test_dataset.select(idxes)
test_dataset = test_dataset.map(speech_file_to_array_fn)

result = test_dataset.map(predict, batched=True, batch_size=20)

label_names = [config.id2label[i] for i in range(config.num_labels)]
label_names = list(map(str,label_names))

y_true = [config.label2id[str(name)] for name in result["label"]]
y_pred = result["predicted"]

print('ground truth of first 10 samples',y_true[:10])
print('prediction of first 10 samples',y_pred[:10])

print(classification_report(y_true, y_pred, target_names=label_names))
