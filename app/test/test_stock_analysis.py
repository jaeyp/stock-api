import json
import matplotlib.pyplot as plt
import pandas as pd
from app.api.divergence import analyze_stock  # Ensure this import matches your project structure

# Load historical stock data from JSON
def load_stock_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

# Main function to generate the graph
def generate_stock_analysis_graph():
    # Load the historical data
    df = load_stock_data('app/data/rdfn_history.json')

    # Ensure the DataFrame has a DateTime index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Extract closing prices
    closing_prices = df['close']
    
    # Prepare to store final scores
    final_scores = []

    # Calculate final scores using analyze_stock for each closing price
    for price in closing_prices:
        score = analyze_stock(price)  # Assuming analyze_stock takes the closing price as input
        final_scores.append(score)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    
    # Plot closing prices
    plt.plot(closing_prices.index, closing_prices, label='Closing Prices', color='blue', marker='o')
    
    # Plot final scores
    plt.plot(closing_prices.index, final_scores, label='Final Scores', color='orange', marker='x')

    # Adding titles and labels
    plt.title('Stock Closing Prices and Final Scores Over the Last Year')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    # Show the plot
    plt.tight_layout()
    plt.show()

# Run the function
if __name__ == "__main__":
    generate_stock_analysis_graph()