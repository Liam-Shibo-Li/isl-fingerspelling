# ISL Fingerspelling Tutor

My final year project (TU856, TU Dublin). The idea is a web app that helps people learn the Irish Sign Language fingerspelling alphabet: you sign a letter to your webcam and the app tells you if it's right, and eventually what to fix.

At the moment this is only the first prototype. It can open the webcam, track the hand and recognise the 23 static letters. There's no feedback, lessons or progress tracking yet.

## Running it

You need Python 3.11 (I used conda):

```
conda create -n isl python=3.11 -y
conda activate isl
pip install mediapipe opencv-python pandas scikit-learn skl2onnx onnx
```

The scripts expect the dataset videos in `data/raw/Videos/Person1` to `Person6` and the MediaPipe model at `ml/models/hand_landmarker.task`. Then:

```
python ml/scripts/extract_landmarks.py     # videos -> data/processed/landmarks.csv
python ml/scripts/train_classifier.py      # compares the models, exports the best one to web/model/
```

To try the web page:

```
cd web
python3 -m http.server 8000
```

and open http://localhost:8000 in Chrome. It needs internet because MediaPipe and ONNX Runtime are loaded from a CDN.

## Data

I'm using the ISL-HS dataset from DCU (Oliveira et al., 2017): https://github.com/marlondcu/ISL

It has 6 people signing every letter 3 times (468 videos). The dataset isn't included here. There's no licence file in the repo, but the authors released it publicly for research, so I'm citing the paper.

MediaPipe found a hand in 29,080 of the 29,104 frames I checked (99.9%). Leaving out J, X and Z (they need movement) gives 26,357 frames for training.

## Models

Each frame gives 21 hand points. Before training I move the wrist to (0, 0) and scale by the size of the palm, so it doesn't matter where the hand is in the picture or how close it is to the camera.

I compared four classifiers using leave-one-signer-out: train on 5 people, test on the one that's left out, and repeat for all 6.

| Model | Accuracy | Macro F1 | ms per frame |
| --- | --- | --- | --- |
| SVM | 94.0% (± 1.8) | 0.934 | 0.47 |
| Logistic Regression | 92.3% (± 3.5) | 0.916 | 0.14 |
| k-NN | 82.7% (± 3.6) | 0.815 | 1.25 |
| Random Forest | 78.8% (± 6.6) | 0.774 | 13.59 |

I went with the SVM. It scored best, it was the most consistent between people (91.2% on the worst person, 97.0% on the best), and it's fast and small enough to run in the browser after exporting to ONNX.

The letters it mixes up the most are G and F (415 frames of G were predicted as F), N and M (319), R and U (204), M and N (187) and C and O (108). The confusion matrix is in `ml/results/confusion_matrix.png`.

## How I got here (including the wrong turns)

**Images or hand points?** My first thought was to train on the video frames directly, since that's what most sign language recognition examples online do. But with only 6 people, a model looking at whole images would probably learn backgrounds, lighting and skin tone as much as hand shape. Holmes et al. (2022) found hand keypoints worked better than raw images for ISL, and I need MediaPipe in the browser anyway, so I switched to using the 21 points.

**Frames or videos?** ISL-HS comes with ready-made frames, so I was going to use those. When I opened them they turned out to be greyscale with the background removed, which looks nothing like a normal webcam image. I went back to the original colour videos.

**Every frame?** At 30 frames a second, neighbouring frames are almost the same, so I only used every second frame. Extraction still took around an hour and a half on my laptop.

**Setup problems.** VS Code was using Python 3.14, which MediaPipe doesn't seem to support yet, so I made a separate conda environment with 3.11.

**Which model?** I expected Random Forest to do well because it's often a strong default, but it was the worst and by far the slowest. Looking at it more, each split only looks at one coordinate, while telling letters apart depends on how several points relate to each other, and that changes from person to person. k-NN had a similar problem with new signers.

**Deep learning?** I thought about it, but with only 6 signers the problem isn't the number of frames, it's the number of different people, and that's too little data to train a deep model on. So I didn't go down that route. MediaPipe is already a deep learning model doing the hard part.

**Testing it on myself.** This is where things looked a lot less good than the numbers. See below.

## What doesn't work well yet

The biggest problem is that it's a lot worse when I actually use it than the 94% suggests. It often gets letters wrong when I try them myself. I think there are a few reasons. The model has only seen 6 people, and its mistakes are pretty consistent for each person, so it's not just noise. My webcam, lighting and angle are different from the dataset. I'm also still learning the alphabet, so my handshapes probably aren't as clean as the signers in the videos. And the test is done per frame, where frames from the same video are almost identical, which probably makes the number look better than it is.

It also always guesses something. As soon as my hand comes into view it shows a letter, even if I'm not making one or I'm moving between letters. The model only knows the 23 letters and has no way of saying "I don't know". Right now low confidence (under 60%) just turns the letter grey.

Some other things:

- G/F, M/N and R/U are still the worst when I test them live, same as in the results.
- It only works with the right hand. Everyone in the dataset is right-handed and I haven't added mirroring for left-handed users. (The page shows my right hand as "Left" because the camera image isn't flipped before MediaPipe sees it, only the display is.)
- No J, X or Z yet.
- The SVM wasn't tuned, I just used C=10.
- When training, scikit-learn 1.9 warns that the way I get confidence scores from the SVM (`probability=True`) will be removed in a future version. It works for now, but I'll need to switch to `CalibratedClassifierCV` later.
- The training landmarks come from the Python version of MediaPipe and the page uses the JavaScript version. They load the same model file, but there might be small differences.

## Getting more data

Since the main limit seems to be that the model has only seen 6 people, the next thing I want to do is find more data. What I've looked at so far:

- **Other ISL datasets.** As far as I can tell, ISL-HS is the only public dataset of separate ISL letters. Signs of Ireland (Trinity) is much bigger, but it's continuous signing rather than single letters and you have to ask for access. It might still be worth asking about. The other "Irish" datasets I found online seem to be smaller copies of ISL-HS, so they're the same 6 people.
- **ASL datasets.** There's a lot more American Sign Language data, but several letters are signed differently in ISL (G, K, L, P and Q, and X moves in ISL but not in ASL). Mixing it in could teach the model the wrong handshapes. I'd only consider letters that I can confirm are the same in both, and even then it's risky.
- **Recording my own.** This is probably the most realistic option: classmates and friends signing the static letters, with the GDPR consent form. It would also give me people who aren't in the training data for testing, and I could record some random hand shapes for an "unknown" class.

If I can't get much more data, I'll keep going with ISL-HS. It's good enough for a working tutor, but it won't be perfect. Some letters will still be confused and it will work better for some people than others. In that case I'd rather the app be honest about it: show "Unknown" when it isn't sure instead of guessing, and report the live accuracy properly in the evaluation rather than just the 94%.

## Next

- Show "Unknown" when the model isn't sure or the prediction keeps jumping around, for example by only showing a letter when the confidence is high enough and the same letter has been predicted for several frames in a row.
- Try to get more data (see above).
- Test it properly with people who aren't in the training data and note the accuracy for each letter.
- Try finger angles as features, which I'll need for the feedback part anyway.
- Start on the feedback, beginning with G/F, M/N and R/U.
- Make a 3D hand model that demonstrates every letter from A to Z, including the movement for J, X and Z. Being able to see the handshape from different angles, and watch the moving letters, should make learning more intuitive than a flat picture or video.

## Credits

- ISL-HS dataset: Oliveira, M. et al. (2017) 'A dataset for Irish Sign Language recognition', IMVIP 2017.
- Holmes, R. et al. (2022) 'Improving signer independent sign language recognition for low resource languages', SLTAT 7, LREC 2022.
- Fowley, F. and Ventresque, A. (2021) 'Sign language fingerspelling recognition using synthetic data', AICS 2021.
- MediaPipe Hands (Zhang et al., 2020), scikit-learn, skl2onnx and ONNX Runtime Web.

I used Claude (Anthropic) to help write parts of the code. The prompts are listed in `docs/ai-prompts.md`.