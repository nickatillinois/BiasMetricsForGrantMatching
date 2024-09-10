import os

def collect_file_names():
    file_names = []
    
    # Iterate through folders 1, 2, 3, and 4
    for folder in ['1', '2', '3', '4']:
        for file in os.listdir(folder):
            # Check if 'ipynb' is not in the file name
            if 'ipynb' not in file:
                # Add .csv to the end of the filename
                #file_with_csv = file + '.csv'
                file_names.append(os.path.join(folder, file))
    
    return file_names

csv_files = collect_file_names()

def generate_mapping(model_paths):
    mapping = {}
    for path in model_paths:
        # Extract model type
        model_type = ''
        if 'roberta-base' in path and not 'xlm-roberta-base' in path:
            model_type = 'roberta'
        elif 'xlm-roberta-base' in path:
            model_type = 'xlm-roberta'
        elif 'spanbert' in path:
            model_type = 'spanbert'
        elif 'albert' in path:
            model_type = 'albert'
        elif 'scibert' in path:
            model_type = 'scibert'
        elif 'bluebert' in path:
            model_type = 'bluebert'
        elif 'distilbert' in path:
            model_type = 'distilbert'
        elif 'biobert' in path:
            model_type = 'biobert'
        elif 'bert-base-multilingual-uncased' in path:
            model_type = 'bert-multi'
        elif 'bert-base-uncased' in path:
            model_type = 'bert-base'
        elif 'bert_uncased_L-12_H-768_A-12' in path:
            model_type = 'bert-base(para)'
        elif 'bert_uncased_L-2_H-128_A-2' in path:
            model_type = 'bert-xxxxx'
        elif 'bert_uncased_L-2_H-256_A-4' in path:
            model_type = 'bert-xxxx'
        elif 'bert_uncased_L-4_H-256_A-4' in path:
            model_type = 'bert-xxx'
        elif 'bert_uncased_L-4_H-512_A-8' in path:
            model_type = 'bert-xx'
        elif 'bert_uncased_L-6_H-512_A-8' in path:
            model_type = 'bert-x'
        elif 'bert_uncased_L-8_H-768_A-12' in path:
            model_type = 'bert-s'
        elif 'electra' in path:
            model_type = 'electra'
        elif 'BiomedBERT' in path:
            model_type = 'biomedbert'
        elif 'deberta' in path:
            model_type = 'deberta'
        
        # Extract dataset
        dataset = ''
        if 'reality' in path:
            dataset = 'r'
        elif 'original' in path:
            dataset = 'o'
        elif 'perfect' in path:
            dataset = 'p'
        
        # Check if names are included
        names_included = 'wi' if 'withoutNames' in path else 'w'
        
        # Create short name
        short_name = f"{model_type}_{dataset}_{names_included}.csv"
        
        # Add to mapping
        mapping[path] = short_name
    
    return mapping

# Generate the mapping
mapping = generate_mapping(csv_files)

# Apply the mapping to rename the files
for original_path, new_name in mapping.items():
    # Split the original path to get the folder and file name separately
    folder, original_file = os.path.split(original_path)
    new_path = os.path.join(folder, new_name)
    
    # Rename the file
    os.rename(original_path, new_path)

# Print the mapping
for original, short in mapping.items():
    print(f"{original} -> {short}")
