import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

# First, manually download the model once
import huggingface_hub
model_path = huggingface_hub.snapshot_download("speechbrain/spkrec-ecapa-voxceleb")

# Load the model from the downloaded path
verification = SpeakerRecognition.from_hparams(
    source=model_path,
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Perform verification
score, prediction = verification.verify_files(
    "recording_t2.wav",
    "recording_r1.wav"
)

print(f"Similarity Score: {score}")
print(f"Same Speaker: {prediction}")