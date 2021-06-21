# sentiment-analysis
NLP sentiment analysis

## contents

- code files
    - main.py : Code that will actually run on the server
    - main.ipynb : A file showing the process of building a neural network model
    
- dataset files
    - dataset/ratings_*.txt : Naver movie review dataset
    - dataset/recent_comment_1000.xlsx : 1000 dataset labeled with real data (used for transfer learning)
    - dataset/comment_labeled.CSV : 200 dataset labeled with real data (used for transfer learning test)

- others
    - best_model.h5 : Trained model with Naver movie review data
    - best_transfer_model.h5 : Transfer trained model with real data
    - requirements.txt : Dependency specification file
    - *.pickle : Pickle files to facilitate neural network training

## env
`pip install -r requirements.txt`