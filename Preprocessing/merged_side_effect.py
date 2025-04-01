import pandas as pd

# Load the data (CSV file)
df = pd.read_csv(r'C:\\Users\\jignesh01\\Desktop\\MedAI\\Data\\combinedDataset_categorized.csv')

# 1. Combine 'substitute0' to 'substitute4' into a single 'substitutes' column
df['substitutes'] = df[['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4']].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

# 2. Combine side effect columns into one 'sideeffects' column
side_effect_columns = [
    'Cardiovascular', 'Gastrointestinal', 'Dermatological', 'Central Nervous System',
    'Allergic and Immunological', 'Hematological', 'Musculoskeletal', 'Respiratory', 
    'Urinary and Renal', 'Endocrine and Metabolic'
]
df['sideeffects'] = df[side_effect_columns].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

# 3. Combine 'short_composition1' and 'short_composition2' into one 'composition' column
df['composition'] = df[['short_composition1', 'short_composition2']].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

# 4. Combine 'use0' to 'use4' into a single 'usecases' column
df['usecases'] = df[['use0', 'use1', 'use2', 'use3', 'use4']].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

# 5. Remove the original individual columns after merging
df = df.drop(columns=['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4'] + 
               side_effect_columns + ['short_composition1', 'short_composition2'] +
               ['use0', 'use1', 'use2', 'use3', 'use4'])

# The final dataframe now has the merged columns
df.to_csv('merged_file.csv', index=False)  # Save the result to a new CSV file
