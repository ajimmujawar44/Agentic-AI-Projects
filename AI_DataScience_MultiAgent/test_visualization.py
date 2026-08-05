from agents.visualization_agent import plot_histogram, plot_correlation_heatmap

csv_path = "uploads/amazon_laptops.csv"   # Change if your file name is different

print(plot_histogram(csv_path, "Price"))
print(plot_correlation_heatmap(csv_path))