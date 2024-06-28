import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def construct_ground_truth(dc_row, dd, cc, dc):
    sim_mutation1 = cc.loc[(cc['mu1'] == dc_row['mu']), 'mu2'].to_list()
    sim_mutation2 = cc.loc[(cc['mu2'] == dc_row['mu']), 'mu1'].to_list()
    similar_mutation = sim_mutation1 + sim_mutation2
    print('len(similar_mutation)',len(similar_mutation))

    sim_drugs1 = dd.loc[(dd['drug1'] == dc_row['drug']), 'drug2'].to_list()
    sim_drugs2 = dd.loc[(dd['drug2'] == dc_row['drug']), 'drug1'].to_list()
    similar_drugs = sim_drugs1 + sim_drugs2
    print('len(similar_drugs)', len(similar_drugs))

    # Create a set to keep track of added combinations
    added_combinations = set()

    # Create a DataFrame to hold g round truth data
    ground_truth_data = []


    for mutation in similar_mutation:
        matched_entries = dc.loc[(dc['drug'] == dc_row['drug']) & (dc['mu'] == mutation) & (dc['rel'] == dc_row['rel'])]
        for _, matched_row in matched_entries.iterrows():
            ground_truth_data.append((matched_row['mu'],matched_row['rel'],dc_row['drug']))
            added_combinations.add((matched_row['drug'], dc_row['mu']))


    for drug in similar_drugs:
        matched_entries = dc.loc[(dc['drug'] == drug) & (dc['mu'] == dc_row['mu']) & (dc['rel'] == dc_row['rel'])]
        for _, matched_row in matched_entries.iterrows():
            ground_truth_data.append((dc_row['mu'],matched_row['rel'],matched_row['drug']))
            added_combinations.add((matched_row['drug'], dc_row['mu']))


    for drug in similar_drugs:
        for mutation in similar_mutation:
            matched_entries = dc.loc[(dc['drug'] == drug) & (dc['mu'] == mutation) & (dc['rel'] == dc_row['rel'])]
            for _, matched_row in matched_entries.iterrows():
                ground_truth_data.append((matched_row['mu'],matched_row['rel'],matched_row['drug']))
                added_combinations.add((matched_row['drug'], matched_row['mu']))


    for drug in similar_drugs:
        if any(dc.loc[(dc['drug'] == drug) & (dc['mu'] == dc_row['mu']), 'rel'] == dc_row['rel']):
            combination = (dc_row['drug'], drug)
            if combination not in added_combinations:
                ground_truth_data.append((dc_row['drug'], 2,drug))  # 2 indicates similarity
                added_combinations.add(combination)

        for mutation in similar_mutation:
            if any(dc.loc[(dc['drug'] == drug) & (dc['mu'] == mutation), 'rel'] == dc_row['rel']):
                combination = (dc_row['drug'], drug)
                if combination not in added_combinations:
                    ground_truth_data.append((dc_row['drug'], 2, drug))  # 2 indicates similarity
                    added_combinations.add(combination)

    # Check for similar mutation relationships with original and similar drugs
    for mutation in similar_mutation:
        # Check against original drug
        if any(dc.loc[(dc['drug'] == dc_row['drug']) & (dc['mu'] == mutation), 'rel'] == dc_row['rel']):
            combination = (dc_row['mu'], mutation)
            if combination not in added_combinations:
                ground_truth_data.append((dc_row['mu'], 3 ,mutation))  # 3 indicates similarity
                added_combinations.add(combination)
        # Check against similar drugs
        for drug in similar_drugs:
            if any(dc.loc[(dc['drug'] == drug) & (dc['mu'] == mutation), 'rel'] == dc_row['rel']):
                combination = (dc_row['mu'], mutation)
                if combination not in added_combinations:
                    ground_truth_data.append((dc_row['mu'], 3, mutation))  # 3 indicates similarity
                    added_combinations.add(combination)

    return ground_truth_data

def load_and_preprocess_data():
    dc = pd.read_csv('../datasets/prediction_datasets/triplets_dc.csv')
    # dc = dc.sample(frac=0.008, random_state=42)
    cc = pd.read_csv('../datasets/prediction_datasets/mu_similar0.97.csv')
    dd = pd.read_csv('../datasets/prediction_datasets/drug_similar0.78.csv')

    dc.columns = ['mu', 'rel', 'drug']
    cc.columns = ['mu1', 'rel', 'mu2']
    dd.columns = ['drug1', 'rel', 'drug2']
    return dc, cc, dd
def process_chunk(chunk, dd, cc, dc):
    ground_truth_rows = []
    for _, row in tqdm(chunk.iterrows(), total=chunk.shape[0]):
        ground_truth = construct_ground_truth(row, dd, cc, dc)
        flat_ground_truth = [item for sublist in ground_truth for item in sublist]
        ground_truth_rows.append([row['mu'], row['rel'], row['drug']] + flat_ground_truth)
    return ground_truth_rows
def main():
    dc, cc, dd = load_and_preprocess_data()

    num_chunks = 5
    chunks = np.array_split(dc, num_chunks)

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_chunk, chunks, [dd]*num_chunks, [cc]*num_chunks, [dc]*num_chunks), total=num_chunks))

    ground_truth_rows = []
    for result in results:
        ground_truth_rows.extend(result)

    max_columns = max(len(row) for row in ground_truth_rows)
    column_names = ['obj', 'rel', 'sbj'] + [f'GT_{i}' for i in range(1, max_columns-2)]

    ground_truth_df = pd.DataFrame(ground_truth_rows, columns=column_names)
    ground_truth_df.to_csv('../datasets/explanation_datasets/gt_all.csv', index=False)

if __name__ == "__main__":
    main()


