

# stock_movements_prediction
The Stock Movement Prediction Project analyzes Reddit sentiment to forecast stock trends. Using BERT for sentiment analysis and machine learning models, it predicts stock movements from social discussions, helping investors make informed decisions.


# Stock-Movement-Prediction
Stock Moment Prediction by Scrapping the data from the Reddit

# Stock Movement Prediction Project

This project uses Reddit data and historical stock information to predict stock price movements using a BERT-based model.

## Prerequisites

- Python 3.7+
- pip (Python package manager)
- CUDA-capable GPU (recommended for faster training)

## Setup

1. Clone the repository:
   
   git clone: https://github.com/S12345-12345/stock_movements_prediction.git
   

Create a virtual environment (optional but recommended):
   
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   

3. Install the required packages:
   
   pip install -r requirements.txt
   

## Dependencies

The main dependencies for this project are:

- pandas
- yfinance
- scikit-learn
- torch
- transformers
- tqdm
hese are all listed in the requirements.txt file and will be installed when you run the pip install command above.

## Data

The project expects a CSV file containing Reddit data. The file should be named reddit_stock_data_20240921_225227.csv and placed in the /kaggle/input/50kcomments/ directory. If your file has a different name or location, update the file path in the main.py script.

## Running the Code

To run the stock prediction model:

1. Ensure you're in the project directory and your virtual environment is activated (if you're using one).

2. Run the main script:
   
   python main.py
   
The script will perform the following steps:
- Preprocess the Reddit data
- Analyze sentiment of the Reddit posts
- Fetch historical stock data for AAPL
- Create features by merging Reddit and stock data
- Train a BERT-based model to predict stock movements
- Evaluate the model and print a classification report

## Customization

- To predict for a different stock, change the stock symbol in the main() function where it calls get_stock_data('AAPL', ...).
- Adjust the num_epochs, batch_size, or learning rate in the train_model() function to optimize performance.
- Modify the features used for prediction by changing the columns selected in the main() function where it defines X.

## Notes

- The code uses CUDA if available. If you don't have a CUDA-capable GPU, it will fall back to CPU, but training will be significantly slower.
- The BERT model requires significant computational resources. If you encounter memory issues, try reducing the batch size or using a smaller BERT model.

## Troubleshooting

- If you encounter a "CUDA out of memory" error, try reducing the batch size in the train_model() function.
- If the code is running too slowly, consider using a subset of your data for initial testing by sampling the dataframe before passing it to the model.

## Contributing
Feel free to fork this repository and submit pull requests with any enhancements.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
