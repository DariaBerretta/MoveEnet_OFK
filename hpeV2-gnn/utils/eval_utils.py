import os, csv

def write_results(path, row):
    # Write a data point into a csvfile
    with open(path, 'a') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(row)


def create_row(skt=[], ts=0.0, delay=0.0):
    # Function to create a row to be written into a csv file.
    # Added dummy ts now. wait for the update ts in processing data
    row = []
    ts = float(ts)
    row.extend([ts, delay])
    row.extend(skt)
    return row


def ensure_loc(path):
    # TODO: add functionality for filenames as well.
    if os.path.isdir(path):
        return True
    else:
        os.makedirs(path)

def create_row_time_graph_making(sample_name, No_of_graph, delay=0.0):
    # Function to create a row to be written into a csv file.
    # Added dummy ts now. wait for the update ts in processing data
    row = []
    No_of_graph = int(No_of_graph)
    row.extend([sample_name, No_of_graph])
    row.append(delay)
    return row