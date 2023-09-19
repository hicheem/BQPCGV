# Blind Quality Predictor for Cloud Gaming Videos (BQPCGV)

## Deep Learning based model Quality metric for Cloud Gaming Content


CNN-based model that accurately assesses the quality of videos in the cloud gaming domain. 
The process consists of three essential phases: Pre training, Fine-tuning, and Video Pooling Quality Prediction Phase.

- The first phase involves pre-training the model using a constructed dataset labeled with the full reference metric LPIPS[^1].
- In the second phase, the model is fine-tuned using Real MOS scores obtained from the GISET[^2] dataset.
- Finally, the frame-level scores are pooled together in the final phase to calculate the overall video score.

<img src="images/proposal.drawio.svg" alt="Comprehensive Overview of the Proposed Approach" width="900">
<!-- <img src="images/video_phase.png" alt="Comprehensive Overview of the Proposed Approach: Exploring The Video Pooling Quality Prediction Stage" width="500"> -->
<!-- ![Comprehensive Overview of the Proposed Approach: Exploring the Initial Two Stages](images/phase1_2.png) -->
<!-- ![Comprehensive Overview of the Proposed Approach: Exploring The Video Pooling Quality Prediction Stage](images/video_phase.png) -->

This repository hosts the code for the complete development process of the three stages. It also provides testing capabilities tailored to your unique use case.

## Table of Contents

- [Requirements](#requirements)
- [How To Use](#how-to-use)
- [Performance Benchmark](#performance-benchmark)
- [References](#references)

## Requirements
```python
pip install -r requirements.txt
```

## How To Use

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

## Performance Benchmark

The models NR-GVQM, NR-GVSQI, NDNetGaming are described in [^3]

##### GVSET [^4]:


|    Models   |SRCC            | PLCC            | KRCC        |
|:------------:|:---------------------:|:--------------------:|:-------------------:|
| NR-GVQM      | 0.870       | 0.890          | -   |
| NR-GVSQI     | 0.860   | 0.870       | -     |
| NDNetGaming   | 0.933  | 0.934 | - |
| **BQPCGV**      |**0.952** | **0.954** | **0.828** |


##### KUGVD [^5]:


|    Models   |SRCC            | PLCC            | KRCC        |
|:------------:|:---------------------:|:--------------------:|:-------------------:|
| NR-GVQM      | 0.910       | 0.910          | -   |
| NR-GVSQI      | 0.880   | 0.890       | -     |
| NDNetGaming      | 0.929  | 0.934 | - |
| **BQPCGV**      |**0.932** | **0.937** | **0.795** |



## References

[^1]: @Zhang, Richard, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. "The unreasonable effectiveness of deep features as a perceptual metric." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8756-8765. 2018.

[^2]: Utke, Markus, Saman Zadtootaghaj, Steven Schmidt, Sebastian Bosse, and Sebastian Möller. "NDNetGaming - Development of a No-Reference Deep CNN for Gaming Video Quality Prediction." In Multimedia Tools and Applications, vol. 79, no. 17, pp. 15003–15024. Springer Nature Switzerland, 2020.

[^3]: Zadtootaghaj (2022). Quality of Experience Modeling for Cloud Gaming Services. Springer.

[^4]: Barman et al. (2018). GamingVideoSET: a dataset for gaming video streaming applications. In 2018 16th Annual Workshop on Network and Systems Support for Games (NetGames), pages 1-6. IEEE.

[^5]: Barman et al. (2019). No-reference video quality estimation based on machine learning for passive gaming video streaming applications. IEEE Access, 7, 74511-74527. IEEE.
