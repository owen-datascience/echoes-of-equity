import pandas as pd

# 1. Define file paths
abundance_path = r"C:\Project-Bioinformatics-Claire\Projects\Breast_cancer_project\PRJNA658160\kraken_G_abundance_matrix.csv"
metadata_path = r"C:\Project-Bioinformatics-Claire\Projects\Breast_cancer_project\PRJNA658160\SraRunTable.csv"

cancer_output_path = r"C:\Project-Bioinformatics-Claire\Projects\Breast_cancer_project\PRJNA658160\cancerous_abundance_matrix.csv"
control_output_path = r"C:\Project-Bioinformatics-Claire\Projects\Breast_cancer_project\PRJNA658160\noncancerous_abundance_matrix.csv"

# 2. Load the data correctly
# We set index_col=0 because the first column contains the SRR row names
abundance_df = pd.read_csv(abundance_path, index_col=0)
metadata_df = pd.read_csv(metadata_path)

# 3. Get the lists of SRR IDs for Case and Control from the 'Run' column
cancer_srrs = metadata_df[metadata_df["case_control"] == "Case"]["Run"].tolist()
control_srrs = metadata_df[metadata_df["case_control"] == "Control"][
    "Run"
].tolist()

# 4. Filter the abundance matrix by rows matching the SRR lists
cancer_df = abundance_df[abundance_df.index.isin(cancer_srrs)]
control_df = abundance_df[abundance_df.index.isin(control_srrs)]

# 5. Save the split data to new CSV files
cancer_df.to_csv(cancer_output_path)
control_df.to_csv(control_output_path)

print("Files successfully split!")
print(
    f"Cancerous matrix saved with {cancer_df.shape[0]} samples and {cancer_df.shape[1]} bacteria columns."
)
print(
    f"Noncancerous matrix saved with {control_df.shape[0]} samples and {control_df.shape[1]} bacteria columns."
)