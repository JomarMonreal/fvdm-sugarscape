import json
import os

def calculate_ratios():
    behavioral_path = "results/visualizations/tables/comparison_behavioral.csv"
    societal_path = "results/visualizations/tables/comparison_societal.csv"
    
    if not os.path.exists(behavioral_path) or not os.path.exists(societal_path):
        print("CSV files not found.")
        return

    import csv
    
    societal_data = {}
    with open(societal_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            societal_data[row["Condition"]] = float(row["Mean Pop"])

    ratios = []
    with open(behavioral_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row["Condition"]
            pop = societal_data.get(cond, 1.0)
            if pop == 0: pop = 1.0
            
            ratios.append({
                "Condition": cond,
                "Rep/Agent": round(float(row["Reproductions"]) / pop, 2),
                "Trade/Agent": round(float(row["Trades"]) / pop, 2),
                "Loan/Agent": round(float(row["Loans"]) / pop, 2),
                "Combat/Agent": round(float(row["Combats"]) / pop, 2)
            })

    # Output as markdown table
    print("| Condition | Rep/Agent | Trade/Agent | Loan/Agent | Combat/Agent |")
    print("| --- | --- | --- | --- | --- |")
    for r in ratios:
        print(f"| {r['Condition']} | {r['Rep/Agent']} | {r['Trade/Agent']} | {r['Loan/Agent']} | {r['Combat/Agent']} |")

if __name__ == "__main__":
    calculate_ratios()
