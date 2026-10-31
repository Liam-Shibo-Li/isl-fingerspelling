# AI prompts used in this project

I used Claude (Anthropic) while building Prototype 1. The main things I asked about are listed below.

**The code.** I asked Claude to help write part of code. I read through the code and ran it myself.

**Understanding the models.** I asked Claude to explain the differences between k-NN, SVM, Random Forest and Logistic Regression, and how to set up a proper comparison using leave-one-signer-out cross-validation.

**Machine learning concepts.** I asked about key ideas like normalising the landmarks, what macro F1 means and why the number of signers matters more than the number of frames.

**ASL data.** I asked whether mixing in ASL fingerspelling data would improve the model or cause problems. Claude explained that several ISL letters are signed differently in ASL, so it could teach the model the wrong handshapes.

**Python version error.** I ran into errors because VS Code was using Python 3.14, which MediaPipe does not support. I asked Claude why this was happening and how to fix it by creating a Python 3.11 environment in conda.

**Improving the 93.7% accuracy.** I asked what could be done to push the accuracy higher. Claude explained that with only 6 signers the main limit is the number of people the model has seen, not the number of frames, so more data would help more than tuning.

**Why live accuracy is lower.** I asked why the model works well on the dataset but often gets letters wrong when I test it myself. Claude explained the likely reasons: different lighting and camera angle, my handshapes as a beginner not matching the dataset, and the test being done per frame where frames from the same video are almost identical.

**Setting up "Unknown".** I asked how to make the model show "Unknown" instead of always guessing a letter. Claude suggested only showing a result when the confidence is high enough and the same letter has been predicted for several frames in a row.