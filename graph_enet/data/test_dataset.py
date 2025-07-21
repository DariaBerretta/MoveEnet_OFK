from scarfDataset import scarfDataset

# events file path
dataset = scarfDataset(root="/home/dberretta-iit.local/data/new_scarfGNN")

print(len(dataset))       # → Number of graphs
graph0 = dataset[0]       # → First graph (PyG `Data` object)

print("DONE")