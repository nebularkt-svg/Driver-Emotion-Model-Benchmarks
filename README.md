# CalmWheel Driver Emotion Model Benchmarks

A comparison archive for five facial expression recognition approaches used in CalmWheel. It contains preprocessing and training scripts, model inputs and configurations, and the available training logs and plots.

[Compare the five approaches](#models) · [Inspect the archived results](results/) · [Run an experiment](#running-the-experiments) · [View the webcam demo](https://github.com/nebularkt-svg/Driver-Emotion-recognition)

## Models

| Experiment | Input | Implementation |
| --- | --- | --- |
| Landmark EmoNet | MediaPipe face landmarks | [Dual-stream landmark model](TrainVla/train.py) |
| Blendshape MLP | MediaPipe blendshape scores | [MLP variant](TrainVla/train2.py) |
| Blendshape attention | MediaPipe blendshape scores | [Attention variant](TrainVla/train2.py) |
| Geometric MLP | Geometry derived from 61 face landmarks | [Feature extraction and classifier](TrainVla/train_geo.py) |
| MobileNetV2 | Face images from AffectNet and RAF-DB | [Image baseline](TrainVla/train_image.py) |

The [archived results](results/) include training curves, logs and available metrics. The geometry, blendshape MLP and attention logs record final-epoch training accuracy of 67.81%, 49.34% and 53.33%, respectively, with no validation result in those epochs. The image log reports mean inference latency of 4.04 ms across 200 timed runs after 20 warm-ups. Its accuracy fields should be read with the original run context, as the log records no validation for the final epoch and the weighted F1 field is zero. These records do not establish an apples-to-apples held-out ranking.

## Running the experiments

Install the Python dependencies and prepare the dataset inputs using the [preprocessing scripts](Dataset/). Landmark runs expect the generated `.npz` data in `facemesh_dataset/`. The image baseline accesses dataset sources through Hugging Face or uses a local dataset path via `--datasets`. Obtain any required dataset access and a MediaPipe Face Landmarker model before preprocessing.

```sh
python -m pip install -r requirements.txt
python TrainVla/run_all_benchmarks.py --quick --only geo_61_au
```

The repository includes training and preprocessing source, archived logs and plots. Dataset images, checkpoints and the separate live-camera application are outside this benchmark snapshot.
