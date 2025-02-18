import json
import matplotlib.pyplot as plt
import numpy as np



def get_sort_order(filename):
    """
    Returns a tuple that defines a custom sort order for the dataset filenames
    based on Japanese proficiency level.
    
    Order breakdown (from easiest to hardest):
      1. JLPT_N5
      2. JLPT_N4
      3. Genki_Vol1
      4. Genki_Vol2
      5. JLPT_N3
      6. JLPT_N2
    Any file not matching one of these patterns will be ordered last.
    """
    if "JLPT_N5" in filename:
        return (1, filename)
    elif "JLPT_N4" in filename:
        return (2, filename)
    elif "Genki_Vol1" in filename:
        return (3, filename)
    elif "Genki_Vol2" in filename:
        return (4, filename)
    elif "JLPT_N3" in filename:
        return (5, filename)
    elif "JLPT_N2" in filename:
        return (6, filename)
    else:
        return (99, filename)

def visualize_average_scores(jsonl_path):
    # Load records from the JSONL file
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
    
    # Sort the records by our custom Japanese proficiency order
    records.sort(key=lambda record: get_sort_order(record["filename"]))
    
    # Extract dataset names, average scores, and calculate standard deviations
    datasets = [record["filename"] for record in records]
    scores = [record["scores"] for record in records]

    # Create short labels for the datasets
    from collections import OrderedDict
    grouped_scores = OrderedDict()
    for fname, score_list in zip(datasets, scores):
        label = ' '.join(fname.split('_')[:2])
        if label not in grouped_scores:
            grouped_scores[label] = []
        grouped_scores[label].extend(score_list)

    short_labels = list(grouped_scores.keys())
    scores = list(grouped_scores.values())

    print(len(short_labels), len(scores))

    # Create a dot plot with error bars to show the average scores per dataset
    plt.figure(figsize=(10, 6))
    plt.violinplot(scores, showmeans=True)
    plt.xticks(ticks=np.arange(1, len(short_labels) + 1), labels=short_labels, rotation=45, ha="right")
    plt.ylabel("Average Score")
    plt.title("Average Scores per Dataset (Ordered by JP Level)")
    # plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Specify the path to your JSONL file
    visualize_average_scores("GPT-output-data/results.jsonl")
