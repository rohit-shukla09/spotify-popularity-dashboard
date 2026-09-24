import pandas as pd

# Load your 38MB Excel file
df = pd.read_excel('cleaned_dataset_small.xls')

# Take half the data (reduces size to ~19MB)
df_tiny = df.sample(frac=0.5, random_state=42)

# Save as CSV (CSV files are significantly lighter than Excel files)
df_tiny.to_csv('final_data.csv', index=False)
print("Done! File is ready for web upload.")