# Exploring the Limits of Zero Shot Vision Language Models for Hate Meme Detection: The Vulnerabilities and their Interpretations [Accepted at AAAI (ICWSM) 2025]
#### For more details about our paper
Naquee Rizwan, Paramananda Bhaskar, Mithun Das, Swadhin Satyaprakash Majhi, Punyajoy Saha, Animesh Mukherjee : "[Exploring the Limits of Zero Shot Vision Language Models for Hate Meme Detection: The Vulnerabilities and their Interpretations"](https://arxiv.org/abs/2402.12198)

# Abstract
There is a rapid increase in the use of multimedia content in current social media platforms. One of the highly popular forms of such multimedia content are memes. While memes have been primarily invented to promote funny and buoyant discussions, malevolent users exploit memes to target individuals or vulnerable communities, making it imperative to identify and address such instances of hateful memes. Thus social media platforms are in dire need for active moderation of such harmful content. While manual moderation is extremely difficult due to the scale of such content, automatic moderation is challenged by the need of good quality annotated data to train hate meme detection algorithms. This makes a perfect pretext for exploring the power of modern day vision language models (VLMs) that have exhibited outstanding performance across various tasks. In this paper we study the effectiveness of VLMs in handling intricate tasks such as hate meme detection in a ***completely zero-shot setting*** so that there is no dependency on annotated data for the task. We perform thorough prompt engineering and query state-of-the-art VLMs using various prompt types to detect hateful/harmful memes. We further interpret the misclassification cases using a novel superpixel based occlusion method. Finally we show that these misclassifications can be neatly arranged into a typology of error classes the knowledge of which should enable the design of better safety guardrails in future.

------------------------------------------
***Folder Description*** :open_file_folder:	
------------------------------------------
```sh
HateVLMs/
├── data_set/ # Dataset loading and processing
│ ├── dataset_wrapper.py # Wrapper class for unified dataset interface
│ ├── facebook_hateful_meme_dataset.py # Facebook Hateful Memes dataset loader
│ ├── mami_hateful_meme_dataset.py # MAMI dataset loader
│ ├── Harm_C_Dataset.py # Harmful Memes (COVID-19) dataset loader
│ ├── Harm_P_Dataset.py # Harmful Memes (Politics) dataset loader
│ ├── BHM_dataset.py # Bangla Hateful Memes dataset loader
│ └── hinglish_dataset.py # HinGlish Offensive Memes dataset loader

├── gpu
│ ├── __init__.py
│ └── gpu_initializer.py # Select device: cpu or cuda

├── models/ # Model initialization and loading
│ ├── idefics_checkpoint_initializer.py
│ ├── instruct_blip_checkpoint_initializer.py
│ └── llava_checkpoint_initializer.py

├── inference/ # Inference scripts for different models
│ ├── idefics_inference.py
│ ├── instruct_blip_inference.py
│ └── llava_inference.py

├── output
│ ├── baselines
│ └── zero_shot

├── superpixels/ # Superpixel-based occlusion analysis
│ ├── superpixels.py # SLIC superpixel generation
│ └── super_pixel_analysis.py # Occlusion-based interpretation

├── zero_shot_analysis/ # Zero-shot evaluation analysis
│ ├── bertopic.ipynb # BERTopic clustering for error typology
│ └── bertopic_explanations.ipynb # Analysis of model explanations

├── utils/ # Utility functions
│ ├── parser_generalized.py # Generalized output parser
│ └── parser_instructBLIP.py # InstructBLIP-specific output parser

├── perform_inference.py # Main script to run inference across models and datasets
└── README.md # Project documentation

```

------------------------------------------
***Key Components*** 
------------------------------------------

### Dataset Loaders (`data_set/`)
- Implement custom dataset classes for each meme dataset:
  - Facebook Hateful Memes (`facebook_hateful_meme_dataset.py`)
  - MAMI (`mami_hateful_meme_dataset.py`) 
  - HARM-C and HARM-P (`Harm_C_Dataset.py`, `Harm_P_Dataset.py`)
  - Bangla Hateful Memes (`BHM_dataset.py`)
  - HinGlish Offensive Memes (`hinglish_dataset.py`)
- Handle data loading, preprocessing, and batching
- Provide a unified dataset interface (`dataset_wrapper.py`)

### Model Initializers (`models/`)
- Load pretrained checkpoints for:
  - IDEFICS (`idefics_checkpoint_initializer.py`)
  - InstructBLIP (`instruct_blip_checkpoint_initializer.py`) 
  - LLaVA (`llava_checkpoint_initializer.py`)
- Apply quantization and other optimizations

### Inference Scripts (`inference/`)
- Implement model-specific inference logic:
  - IDEFICS (`idefics_inference.py`)
  - InstructBLIP (`instruct_blip_inference.py`)
  - LLaVA (`llava_inference.py`)
- Handle different prompt strategies and input formats

### Superpixel Analysis (`superpixels/`)
- Generate superpixels using SLIC algorithm (`superpixels.py`)
- Perform occlusion-based interpretation of model predictions (`super_pixel_analysis.py`)

### Zero-shot Analysis (`zero_shot_analysis/`)
- Use BERTopic for clustering misclassifications (`bertopic.ipynb`)
- Analyze model-generated explanations (`bertopic_explanations.ipynb`)

### Utility Functions (`utils/`)
- Parse model outputs into standardized formats:
  - General parser (`parser_generalized.py`)
  - InstructBLIP-specific parser (`parser_instructBLIP.py`)
- Handle model-specific quirks in output parsing

### Main Inference Script (`perform_inference.py`)
- Orchestrate the evaluation process across models and datasets
- Apply different prompt strategies and collect results

This project structure enables a systematic evaluation of Vision Language Models (VLMs) for hate meme detection across multiple datasets and languages. It provides tools for zero-shot inference, error analysis, and interpretation of model decisions using superpixel occlusion.

## Contact

For any questions or issues, please contact: nrizwan@kgpian.iitkgp.ac.in, pbhaskar@kgpian.iitkgp.ac.in

Please cite our paper:
~~~bibtex
@misc{rizwan2024zeroshotvlmshate,
      title={Zero shot VLMs for hate meme detection: Are we there yet?}, 
      author={Naquee Rizwan and Paramananda Bhaskar and Mithun Das and Swadhin Satyaprakash Majhi and Punyajoy Saha and Animesh Mukherjee},
      year={2024},
      eprint={2402.12198},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.12198}, 
}
~~~