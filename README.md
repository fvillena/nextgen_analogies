# Clinical analogy resolution performance for foundation language models

Fabián Villena, Tamara Quiroga & Jocelyn Dunstan

Using extensive data sources to create foundation language models has revolutionized the performance of deep learning-based architectures. This remarkable improvement has led to state-of-the-art results for various downstream NLP tasks, including clinical tasks. However, more research is needed to measure model performance intrinsically, especially in the clinical domain. We revisit the use of analogy questions as an effective method to measure the intrinsic performance of language models for the clinical domain in English. We tested multiple Transformers-based language models over analogy questions constructed from the Unified Medical Language System (UMLS), a massive knowledge graph of clinical concepts. Our results show that large language models are significantly more performant for analogy resolution than small language models.
Similarly, domain-specific language models perform better than general domain language models. We also found a correlation between intrinsic and extrinsic performance, validated through PubMedQA extrinsic task. Creating clinical-specific and language-specific language models is essential for advancing biomedical and clinical NLP and will ensure a valid application in clinical practice.
Finally, given that our proposed intrinsic test is based on a term graph available in multiple languages, the dataset can be built to measure the performance of models in languages other than English.

## Prepare the environment

To run the code in this repository, we prepared a `.devcontainer` with all the necessary dependencies. You can use the devcontainer by installing the Remote - Containers extension in Visual Studio Code and opening the repository in a container. For more information, see [Developing inside a Container](https://code.visualstudio.com/docs/remote/containers).

## Building the analogy dataset

1. Query the UMLS database to extract all the concept pairs with the script `analogies.sql`. 
2. Build the analogy questions with the script `build_analogies.py`. The instructions on how to run the script are in the file.
3. Measure the performance on the analogy dataset with the script `measure_analogies.py`. The instructions on how to run the script are in the file.