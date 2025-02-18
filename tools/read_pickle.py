import pickle

def read_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Example usage
file_path = "output/results.pkl"  # Replace with your pickle file path
content = read_pickle(file_path)
print(content)
