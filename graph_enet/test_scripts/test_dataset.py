from graph_enet.data.scarfDataset import scarfDataset
# This script is used to test the scarfDataset class and ensure it works correctly.
# It performs sanity checks on the dataset, ensuring that it is not empty and that the processed files are correctly loaded.


# events file path
dataset = scarfDataset(root="/home/dberretta-iit.local/data/new_scarfGNN")

print("Dataset sanity checks...")
print('==========================')

# Check the length of the dataset
assert len(dataset) > 0, "Dataset is empty!"

# Check if the processed files are correctly loaded
for i in range(len(dataset)):
    data = dataset.get(i)
    assert data is not None, f"Data at index {i} is None!"
    assert hasattr(data, 'edge_index'), f"Data at index {i} does not have 'edge_index' attribute!"
    assert hasattr(data, 'x'), f"Data at index {i} does not have 'x' attribute!"


# Print dataset information
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')        # No need because this is a regression task

data = dataset[0]  # Get the first graph object.

print(f'First graph data: {data}')
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

print("DONE")