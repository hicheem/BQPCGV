# Blind Quality Predictor for Cloud Gaming Videos (BQPCGV)

## Deep Learning based model Quality metric for Cloud Gaming Content


CNN-based model that accurately assesses the quality of videos in the cloud gaming domain. 
The process consists of three essential phases: Pre training, Fine-tuning, and Video Pooling Quality Prediction Phase.

- The first phase involves pre-training the model using a constructed dataset labeled with the full reference metric LPIPS[^1].
- In the second phase, the model is fine-tuned using Real MOS scores obtained from the GISET[^2] dataset.
- Finally, the frame-level scores are pooled together in the final phase to calculate the overall video score.

<img src="images/phase1_2.png" alt="Comprehensive Overview of the Proposed Approach: Exploring the Initial Two Stages" width="500">
<img src="images/video_phase.png" alt="Comprehensive Overview of the Proposed Approach: Exploring The Video Pooling Quality Prediction Stage" width="500">
<!-- ![Comprehensive Overview of the Proposed Approach: Exploring the Initial Two Stages](images/phase1_2.png) -->
<!-- ![Comprehensive Overview of the Proposed Approach: Exploring The Video Pooling Quality Prediction Stage](images/video_phase.png) -->

This repository hosts the code for the complete development process of the three stages. It also provides testing capabilities tailored to your unique use case.

## How to Use

To test the model for a given video, follow these steps:

1. Run the `test.py` script with the following command:

    ```shell
        python test.py 
            --model=./models/model_Final_DMOS.h5 
            --videopath=./videos/ 
            --videoname=video1.mp4 
            --framepersecond=1
    ```

2. Use the following command-line options to configure the test:

- `--model` or `-m`: Specify the path to the model file.
- `--videopath` or `-vp`: Specify the folder path where the video is located.
- `--videoname` or `-vn`: Specify the name of the video file.
- `--framepersecond` or `-fps`: Specify the number of frames to process per second.

3. For more help run:

    ```shell
        python test.py -h
    ```


[^1]: @Zhang, Richard, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. "The unreasonable effectiveness of deep features as a perceptual metric." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8756-8765. 2018.

[^2]: Utke, Markus, Saman Zadtootaghaj, Steven Schmidt, Sebastian Bosse, and Sebastian Möller. "NDNetGaming - Development of a No-Reference Deep CNN for Gaming Video Quality Prediction." In Multimedia Tools and Applications, vol. 79, no. 17, pp. 15003–15024. Springer Nature Switzerland, 2020.