# Exploring the Limits of Zero Shot Vision Language Models for Hate Meme Detection: The Vulnerabilities and their Interpretations

Accepted at **AAAI (ICWSM) 2025**

Naquee Rizwan, Paramananda Bhaskar, Mithun Das, Swadhin Satyaprakash Majhi, Punyajoy Saha, Animesh Mukherjee:
[[Paper]](https://ojs.aaai.org/index.php/ICWSM/article/view/35894)
[[Arxiv]](https://arxiv.org/abs/2402.12198v3)

------------------------------------------
## Abstract
There is a rapid increase in the use of multimedia content in current social media platforms. One of the highly popular forms of such multimedia content are memes. While memes have been primarily invented to promote funny and buoyant discussions, malevolent users exploit memes to target individuals or vulnerable communities, making it imperative to identify and address such instances of hateful memes. Thus social media platforms are in dire need for active moderation of such harmful content. While manual moderation is extremely difficult due to the scale of such content, automatic moderation is challenged by the need of good quality annotated data to train hate meme detection algorithms. This makes a perfect pretext for exploring the power of modern day vision language models (VLMs) that have exhibited outstanding performance across various tasks. In this paper we study the effectiveness of VLMs in handling intricate tasks such as hate meme detection in a ***completely zero-shot setting*** so that there is no dependency on annotated data for the task. We perform thorough prompt engineering and query state-of-the-art VLMs using various prompt types to detect hateful/harmful memes. We further interpret the misclassification cases using a novel superpixel based occlusion method. Finally we show that these misclassifications can be neatly arranged into a typology of error classes the knowledge of which should enable the design of better safety guardrails in future.

------------------------------------------
## File/Folder Description :open_file_folder:
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

This project structure enables a systematic evaluation of Vision Language Models (VLMs) for hate meme detection 
across multiple datasets and languages in zero-shot setup. Further, it provides tools for automatic typology 
induction using BERTopic, error analysis, and **black-box** model interpretation using superpixel occlusion.

------------------------------------------
## Please cite our paper
~~~bibtex
@article{Rizwan_Bhaskar_Das_Majhi_Saha_Mukherjee_2025, title={Exploring the Limits of Zero Shot Vision Language Models for Hate Meme Detection: The Vulnerabilities and their Interpretations}, volume={19}, url={https://ojs.aaai.org/index.php/ICWSM/article/view/35894}, DOI={10.1609/icwsm.v19i1.35894}, abstractNote={There is a rapid increase in the use of multimedia content in current social media platforms. One of the highly popular forms of such multimedia content are memes. While memes have been primarily invented to promote funny and buoyant discussions, malevolent users exploit memes to target individuals or vulnerable communities, making it imperative to identify and address such instances of hateful memes. Thus social media platforms are in dire need for active moderation of such harmful content. While manual moderation is extremely difficult due to the scale of such content, automatic moderation is challenged by the need of good quality annotated data to train hate meme detection algorithms. This makes a perfect pretext for exploring the power of modern day vision language models (VLMs) that have exhibited outstanding performance across various tasks. In this paper we study the effectiveness of VLMs in handling intricate tasks such as hate meme detection in a completely zero-shot setting so that there is no dependency on annotated data for the task. We perform thorough prompt engineering and query state-of-the-art VLMs using various prompt types to detect hateful/harmful memes. We further interpret the misclassification cases using a novel superpixel based occlusion method. Finally we show that these misclassifications can be neatly arranged into a typology of error classes the knowledge of which should enable the design of better safety guardrails in future. Code and other relevant sources are available online.
Warning: Contains potentially offensive content.}, number={1}, journal={Proceedings of the International AAAI Conference on Web and Social Media}, author={Rizwan, Naquee and Bhaskar, Paramananda and Das, Mithun and Majhi, Swadhin Satyaprakash and Saha, Punyajoy and Mukherjee, Animesh}, year={2025}, month={Jun.}, pages={1669-1689} }
~~~

~~~bibtex
@misc{rizwan2025exploringlimitszeroshot,
      title={Exploring the Limits of Zero Shot Vision Language Models for Hate Meme Detection: The Vulnerabilities and their Interpretations}, 
      author={Naquee Rizwan and Paramananda Bhaskar and Mithun Das and Swadhin Satyaprakash Majhi and Punyajoy Saha and Animesh Mukherjee},
      year={2025},
      eprint={2402.12198},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.12198}, 
}
~~~

------------------------------------------
## Contact
For any questions or issues, please contact: nrizwan@kgpian.iitkgp.ac.in, pbhaskar@kgpian.iitkgp.ac.in
