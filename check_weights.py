import json
import numpy as np
import csv

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def check_weights():
    with open("results/fvdm_weights.json", "r") as f:
        models = json.load(f)

    actions = list(models.keys())
    if not actions:
        print("No models found.")
        return

    coords = ["I", "D", "P", "X"]
    output_file = "results/weight_similarities.csv"
    
    print("=== Weight Vector Comparisons ===")
    print("Showing Cosine Similarity [-1 to 1] between action weight vectors.")
    print(f"Results will be saved to {output_file}\n")

    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Coordinate", "Action1", "Action2", "CosineSimilarity"])

        for c in coords:
            print(f"--- Coordinate: {c} ---")
            
            valid_actions = [a for a in actions if c in models[a]]
            
            if len(valid_actions) < 2:
                print(f"Not enough actions have coordinate {c} to compare.")
                continue
                
            matrix = np.zeros((len(valid_actions), len(valid_actions)))
            
            for i, a1 in enumerate(valid_actions):
                w1 = np.array(models[a1][c]["weights"])
                for j, a2 in enumerate(valid_actions):
                    w2 = np.array(models[a2][c]["weights"])
                    sim = cosine_similarity(w1, w2)
                    matrix[i, j] = sim
                    # Write to CSV in a flat format
                    writer.writerow([c, a1, a2, f"{sim:.4f}"])
            
            # Print to terminal as a table
            header = f"{'':>10} | " + " | ".join([f"{a:>8}" for a in valid_actions])
            print(header)
            print("-" * len(header))
            
            for i, a1 in enumerate(valid_actions):
                row_str = f"{a1:>10} | "
                for j, a2 in enumerate(valid_actions):
                    val = matrix[i, j]
                    row_str += f"{val:8.3f} | "
                print(row_str)
            print("\n")
            
    print(f"Successfully saved to {output_file}")

if __name__ == "__main__":
    check_weights()
