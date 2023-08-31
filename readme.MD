# Text Classification using Sentence Embeddings

This project aims to compare various sentence embeddings for text classification. It takes two text files as inputs and classifies sentences/paragraphs from those files using three different models - BAAI, Glove, and TF-IDF. The results are visually represented using different charts to compare the performance of these models.

## Data Sources

We have leveraged the following data sources for this project:

1. **HackerRank** - The challenge titled "Byte The Correct Apple" was applied in our research, offering data to distinguish between Apple Inc. and the apple fruit. The detailed challenge, penned by PRASHANTB1984, is accessible at [HackerRank's Byte The Correct Apple Challenge](https://www.hackerrank.com/challenges/byte-the-correct-apple/problem). In particular, we adopted the text files referencing Wikipedia on both Apple Inc. and apple fruit as outlined in the challenge description.

2. **Kaggle** - Our project also employed the "Animal Mouse vs Computer Mouse" dataset available on Kaggle. This dataset provides essential insights on the differentiation between the computer mouse and its living counterpart. This dataset is shared under the CC0: Public Domain license. For a comprehensive overview or to access the dataset, you can navigate to [Kaggle's Animal Mouse vs Computer Mouse Dataset](https://www.kaggle.com/datasets/werty12121/animal-mouse-vs-computer-mouse-text-dataset?resource=download).

For the record, all data integrated into this project is strictly for academic and research intentions. We urge users to adhere to licensing commitments and terms of service when employing and redistributing the data.


## Getting Started

### Setting Up the Environment

To ensure that all dependencies are met, we provide an environment YAML file named `env.yml`. This can be used to set up a Conda environment named `hf` which contains all the necessary packages.

Run the following command to create the `hf` environment:

```bash
conda env create -f env.yml
```

### File Structure

1. `classify.ipynb`: This Jupyter notebook contains the main code for reading the files, processing the data, training the models, making predictions, and visualizing the results.
2. `models.py`: Contains the classes for different embedding models (BAIModel, GloveModel, TFIDFModel).
3. `utils.py`: Contains utility functions for reading and processing the input files.
4. `charts.py`: Contains functions for plotting performance metrics and confusion matrices.

### Usage

1. Place your text files in the `data` directory.
2. Open the `classify.ipynb` notebook.
3. Add your file pairs to the `file_pairs` list.
4. Run the notebook.

## How It Works

1. Two text files are read and split into training and testing datasets.
2. Each of the three models (BAIModel, GloveModel, and TFIDFModel) computes sentence embeddings for the training data.
3. Each model then classifies the test data based on the embeddings.
4. The predictions are compared with the ground truth to compute performance metrics: Accuracy, Precision, Recall, and F1 Score.
5. The metrics are plotted using bar charts for each model.
6. A confusion matrix is plotted for each model to visualize the classification results.
7. A combined chart is used to compare the performance of all three models.

## Example

For the file pairs:
- "apple-computers.txt" and "apple-fruit.txt"
- "mouse-computers.txt" and "mouse-animal.txt"

The tool will classify sentences or paragraphs from these files using each model and display the results visually.
