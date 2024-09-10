import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

def compute_correlation_matrix_for_subplot(ax, intrinsic_columns, extrinsic_columns, filename='intrinsic_bias.csv', alpha=0.05, custom_labels=None, custom_title='Correlation Matrix'):
   # Read the DataFrame from the CSV file
   df = pd.read_csv(filename)

   # Ensure columns are in list form
   if isinstance(intrinsic_columns, str):
       intrinsic_columns = [intrinsic_columns]
   if isinstance(extrinsic_columns, str):
       extrinsic_columns = [extrinsic_columns]

   # Combine the columns of interest
   columns_of_interest = intrinsic_columns + extrinsic_columns

   # Initialize the Pearson correlation matrix and the annotation matrix
   pearson_corr_matrix = pd.DataFrame(index=columns_of_interest, columns=columns_of_interest)
   annot_matrix = pd.DataFrame(index=columns_of_interest, columns=columns_of_interest, dtype=str)

   # Calculate Pearson correlations and annotate significance
   for col1 in columns_of_interest:
       for col2 in columns_of_interest:
           if col1 != col2:
               pearson_corr, p_value = pearsonr(df[col1], df[col2])
               pearson_corr_matrix.loc[col1, col2] = pearson_corr
               if p_value < alpha:
                   annot_matrix.loc[col1, col2] = f'{pearson_corr:.2f}*'
               else:
                   annot_matrix.loc[col1, col2] = f'{pearson_corr:.2f}'
           else:
               pearson_corr_matrix.loc[col1, col2] = 1.0
               annot_matrix.loc[col1, col2] = '1.00'

   # Apply custom labels if provided
   if custom_labels:
       pearson_corr_matrix.rename(index=custom_labels, columns=custom_labels, inplace=True)
       annot_matrix.rename(index=custom_labels, columns=custom_labels, inplace=True)

   # Mask the upper triangle of the matrix
   mask = np.triu(np.ones_like(pearson_corr_matrix, dtype=bool))

   # Plot the Pearson correlation matrix in the given subplot
   sns.heatmap(pearson_corr_matrix.astype(float), annot=annot_matrix, fmt='', cmap='coolwarm', cbar=True, vmin=-1, vmax=1, mask=mask, linewidths=0.5, square=True, ax=ax)
   ax.set_title(custom_title, fontsize=12, pad=10)
   ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
   ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

   # Increase the font size of all text
   for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
       item.set_fontsize(14)

### Step 3: Create a Combined Figure with 6 Subplots

def create_combined_figure_race():
    # Create a figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(3, 2, figsize=(22, 30))
    plt.subplots_adjust(hspace=0.001)

    compute_correlation_matrix_for_subplot(axes[0, 0],
    intrinsic_columns=['disco_asian','disco_black','disco_hispanic','SEAT_ABW','tan_SEAT_ABW','lau_SEAT_ABW',
        'SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA'],
    extrinsic_columns=['o_wi_bias_a_mean_norm','o_wi_bias_b_mean_norm','o_wi_bias_h_mean_norm','o_wi_bias_fb_mean_norm'],
     custom_labels={
         'disco_asian': 'DisCo (A-W)',
         'disco_black': 'DisCo (B-W)',
         'disco_hispanic': 'DisCo (H-W)',
         'SEAT_ABW': 'May et al. (2019) (ABW)',
         'tan_SEAT_ABW': 'Tan et al. (2019) (ABW)',
         'lau_SEAT_ABW': 'Lauscher et al. (2021) (ABW)',
        'SEAT_EAvsAA': 'May et al. (2019) (W-B)',
        'tan_SEAT_EAvsAA': 'Tan et al. (2019) (W-B)',
        'lau_SEAT_EAvsAA': 'Lauscher et al. (2021) (W-B)',
        'o_wi_bias_a_mean_norm':'A-W_NormDiff@1',
        'o_wi_bias_b_mean_norm':'B-W_NormDiff@1',
        'o_wi_bias_h_mean_norm':'H-W_NormDiff@1',
        'o_wi_bias_fb_mean_norm': 'BF-WF_NormDiff@1',
    },
    custom_title='Original Data without PI Names'
    )
    compute_correlation_matrix_for_subplot(axes[0, 1],
        intrinsic_columns=['disco_asian','disco_black','disco_hispanic','SEAT_ABW','tan_SEAT_ABW','lau_SEAT_ABW',
            'SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA'],
        extrinsic_columns=['o_w_bias_b_mean_norm','o_w_bias_a_mean_norm','o_w_bias_h_mean_norm','o_w_bias_fb_mean_norm'],
        custom_labels={
            'disco_asian': 'DisCo (A-W)',
            'disco_black': 'DisCo (B-W)',
            'disco_hispanic': 'DisCo (H-W)',
            'SEAT_ABW': 'May et al. (2019) (ABW)',
            'tan_SEAT_ABW': 'Tan et al. (2019) (ABW)',
            'lau_SEAT_ABW': 'Lauscher et al. (2021) (ABW)',
            'SEAT_EAvsAA': 'May et al. (2019) (W-B)',
            'tan_SEAT_EAvsAA': 'Tan et al. (2019) (W-B)',
            'lau_SEAT_EAvsAA': 'Lauscher et al. (2021) (W-B)',
            'o_w_bias_a_mean_norm':'A-W_NormDiff@1',
            'o_w_bias_b_mean_norm':'B-W_NormDiff@1',
            'o_w_bias_h_mean_norm':'H-W_NormDiff@1',
            'o_w_bias_fb_mean_norm': 'BF-WF_NormDiff@1',
        },
        custom_title='Original Data with PI Names'
    )
    compute_correlation_matrix_for_subplot(axes[1, 0],
        intrinsic_columns=['disco_asian','disco_black','disco_hispanic','SEAT_ABW','tan_SEAT_ABW','lau_SEAT_ABW',
            'SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA'],
        extrinsic_columns=['r_wi_bias_b_mean_norm','r_wi_bias_a_mean_norm','r_wi_bias_h_mean_norm','r_wi_bias_fb_mean_norm'],
        custom_labels={
            'disco_asian': 'DisCo (A-W)',
            'disco_black': 'DisCo (B-W)',
            'disco_hispanic': 'DisCo (H-W)',
            'SEAT_ABW': 'May et al. (2019) (ABW)',
            'tan_SEAT_ABW': 'Tan et al. (2019) (ABW)',
            'lau_SEAT_ABW': 'Lauscher et al. (2021) (ABW)',
            'SEAT_EAvsAA': 'May et al. (2019) (W-B)',
            'tan_SEAT_EAvsAA': 'Tan et al. (2019) (W-B)',
            'lau_SEAT_EAvsAA': 'Lauscher et al. (2021) (W-B)',
            'r_wi_bias_a_mean_norm':'A-W_NormDiff@1',
            'r_wi_bias_b_mean_norm':'B-W_NormDiff@1',
            'r_wi_bias_h_mean_norm':'H-W_NormDiff@1',
            'r_wi_bias_fb_mean_norm': 'BF-WF_NormDiff@1',
        },
        custom_title='Realistic Data without PI Names'
    )
    compute_correlation_matrix_for_subplot(axes[1, 1],
        intrinsic_columns=['disco_asian','disco_black','disco_hispanic','SEAT_ABW','tan_SEAT_ABW','lau_SEAT_ABW',
            'SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA'],
        extrinsic_columns=['r_w_bias_b_mean_norm','r_w_bias_a_mean_norm','r_w_bias_h_mean_norm','r_w_bias_fb_mean_norm'],
        custom_labels={
            'disco_asian': 'DisCo (A-W)',
            'disco_black': 'DisCo (B-W)',
            'disco_hispanic': 'DisCo (H-W)',
            'SEAT_ABW': 'May et al. (2019) (ABW)',
            'tan_SEAT_ABW': 'Tan et al. (2019) (ABW)',
            'lau_SEAT_ABW': 'Lauscher et al. (2021) (ABW)',
            'SEAT_EAvsAA': 'May et al. (2019) (W-B)',
            'tan_SEAT_EAvsAA': 'Tan et al. (2019) (W-B)',
            'lau_SEAT_EAvsAA': 'Lauscher et al. (2021) (W-B)',
            'r_w_bias_a_mean_norm':'A-W_NormDiff@1',
            'r_w_bias_b_mean_norm':'B-W_NormDiff@1',
            'r_w_bias_h_mean_norm':'H-W_NormDiff@1',
            'r_w_bias_fb_mean_norm': 'BF-WF_NormDiff@1',
        },
        custom_title='Realistic Data with PI Names'
    )
    compute_correlation_matrix_for_subplot(axes[2, 0],
        intrinsic_columns=['disco_asian','disco_black','disco_hispanic','SEAT_ABW','tan_SEAT_ABW','lau_SEAT_ABW',
            'SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA'],
        extrinsic_columns=['p_wi_bias_b_mean_norm','p_wi_bias_a_mean_norm','p_wi_bias_h_mean_norm','p_wi_bias_fb_mean_norm'],
        custom_labels={
            'disco_asian': 'DisCo (A-W)',
            'disco_black': 'DisCo (B-W)',
            'disco_hispanic': 'DisCo (H-W)',
            'SEAT_ABW': 'May et al. (2019) (ABW)',
            'tan_SEAT_ABW': 'Tan et al. (2019) (ABW)',
            'lau_SEAT_ABW': 'Lauscher et al. (2021) (ABW)',
            'SEAT_EAvsAA': 'May et al. (2019) (W-B)',
            'tan_SEAT_EAvsAA': 'Tan et al. (2019) (W-B)',
            'lau_SEAT_EAvsAA': 'Lauscher et al. (2021) (W-B)',
            'p_wi_bias_a_mean_norm':'A-W_NormDiff@1',
            'p_wi_bias_b_mean_norm':'B-W_NormDiff@1',
            'p_wi_bias_h_mean_norm':'H-W_NormDiff@1',
            'p_wi_bias_fb_mean_norm': 'BF-WF_NormDiff@1',
        },
        custom_title='Balanced Data without PI Names'
    )
    compute_correlation_matrix_for_subplot(axes[2, 1],
        intrinsic_columns=['disco_asian','disco_black','disco_hispanic','SEAT_ABW','tan_SEAT_ABW','lau_SEAT_ABW',
            'SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA'],
        extrinsic_columns=['p_w_bias_b_mean_norm','p_w_bias_a_mean_norm','p_w_bias_h_mean_norm','p_w_bias_fb_mean_norm'],
        custom_labels={
            'disco_asian': 'DisCo (A-W)',
            'disco_black': 'DisCo (B-W)',
            'disco_hispanic': 'DisCo (H-W)',
            'SEAT_ABW': 'May et al. (2019) (ABW)',
            'tan_SEAT_ABW': 'Tan et al. (2019) (ABW)',
            'lau_SEAT_ABW': 'Lauscher et al. (2021) (ABW)',
            'SEAT_EAvsAA': 'May et al. (2019) (W-B)',
            'tan_SEAT_EAvsAA': 'Tan et al. (2019) (W-B)',
            'lau_SEAT_EAvsAA': 'Lauscher et al. (2021) (W-B)',
            'p_w_bias_a_mean_norm':'A-W_NormDiff@1',
            'p_w_bias_b_mean_norm':'B-W_NormDiff@1',
            'p_w_bias_h_mean_norm':'H-W_NormDiff@1',
            'p_w_bias_fb_mean_norm': 'BF-WF_NormDiff@1',
        },
        custom_title='Balanced Data with PI Names')
    

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("race_correlation_norm_matrices.png", dpi=1200)
    plt.show()

def create_combined_figure_gender():
    # Create a figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(20,10))
    plt.subplots_adjust(hspace=0.001)
    
    compute_correlation_matrix_for_subplot(axes[0, 0], 
        intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
        extrinsic_columns=['o_wi_bias_f_mean_norm'],
        custom_labels={
            'disco_gender': 'DisCo (Gender)',
            'lpbs_mean': 'LPBS',
            'SEAT_gender': 'May et al. (2019)',
            'tan_SEAT_gender': 'Tan et al. (2019)',
            'lau_SEAT_gender': 'Lauscher et al. (2021)',
            'o_wi_bias_f_mean_norm': 'F-M_NormDiff@1',
        },
        custom_title='Original Data without PI Names'
    )
    compute_correlation_matrix_for_subplot(axes[1, 0], 
        intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
        extrinsic_columns=['o_w_bias_f_mean_norm'],
        custom_labels={
            'disco_gender': 'DisCo (Gender)',
            'lpbs_mean': 'LPBS',
            'SEAT_gender': 'May et al. (2019)',
            'tan_SEAT_gender': 'Tan et al. (2019)',
            'lau_SEAT_gender': 'Lauscher et al. (2021)',
            'o_w_bias_f_mean_norm': 'F-M_NormDiff@1',
        },
        custom_title='Original Data with PI Names'
    )
    compute_correlation_matrix_for_subplot(axes[0, 1], 
        intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
        extrinsic_columns=['r_wi_bias_f_mean_norm'],
        custom_labels={
            'disco_gender': 'DisCo (Gender)',
            'lpbs_mean': 'LPBS',
            'SEAT_gender': 'May et al. (2019)',
            'tan_SEAT_gender': 'Tan et al. (2019)',
            'lau_SEAT_gender': 'Lauscher et al. (2021)',
            'r_wi_bias_f_mean_norm': 'F-M_NormDiff@1',
        },
        custom_title='Realistic Data without PI Names'
    )
    compute_correlation_matrix_for_subplot(axes[1, 1],
        intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
        extrinsic_columns=['r_w_bias_f_mean_norm'],
        custom_labels={
            'disco_gender': 'DisCo (Gender)',
            'lpbs_mean': 'LPBS',
            'SEAT_gender': 'May et al. (2019)',
            'tan_SEAT_gender': 'Tan et al. (2019)',
            'lau_SEAT_gender': 'Lauscher et al. (2021)',
            'r_w_bias_f_mean_norm': 'F-M_NormDiff@1',
        },
        custom_title='Realistic Data with PI Names'
    )
    compute_correlation_matrix_for_subplot(axes[0, 2],
        intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
        extrinsic_columns=['p_wi_bias_f_mean_norm'],
        custom_labels={
            'disco_gender': 'DisCo (Gender)',
            'lpbs_mean': 'LPBS',
            'SEAT_gender': 'May et al. (2019)',
            'tan_SEAT_gender': 'Tan et al. (2019)',
            'lau_SEAT_gender': 'Lauscher et al. (2021)',
            'p_wi_bias_f_mean_norm': 'F-M_NormDiff@1',
        },
        custom_title='Balanced Data without PI Names'
    )
    compute_correlation_matrix_for_subplot(axes[1, 2],
        intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
        extrinsic_columns=['p_w_bias_f_mean_norm'],
        custom_labels={
            'disco_gender': 'DisCo (Gender)',
            'lpbs_mean': 'LPBS',
            'SEAT_gender': 'May et al. (2019)',
            'tan_SEAT_gender': 'Tan et al. (2019)',
            'lau_SEAT_gender': 'Lauscher et al. (2021)',
            'p_w_bias_f_mean_norm': 'F-M_NormDiff@1',
        },
        custom_title='Balanced Data with PI Names'
    )

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("gender_correlation_norm_matrices.png",dpi=1200)

    plt.show()


def calculate_correlations(column1, column2, filename='intrinsic_bias.csv', alpha=0.05):
    # Read the DataFrame from the CSV file
    df = pd.read_csv(filename)
    
    # Check if the columns exist in the DataFrame
    if column1 not in df.columns:
        print(f'Column "{column1}" does not exist in the DataFrame.')
        return
    if column2 not in df.columns:
        print(f'Column "{column2}" does not exist in the DataFrame.')
        return
    
    # Pearson correlation
    pearson_corr, pearson_p_value = pearsonr(df[column1], df[column2])
    
    # Determine if p-values are significant
    pearson_significant = pearson_p_value < alpha
    
    # Print the correlations and significance
    print(f'Pearson correlation between "{column1}" and "{column2}": {pearson_corr:.4f}')
    print(f'Pearson p-value: {pearson_p_value:.4f}')
    print(f'Pearson correlation is {"significant" if pearson_significant else "not significant"} at alpha={alpha}')

df = pd.read_csv('intrinsic_bias.csv')
chosen_data = 'o'
names_included = 'wi'
gender_or_race = 'f'
average_or_median = 'mean'
bias_type = 'desc' #('desc' of 'norm' )
intrinsic_metric = 'disco_gender'
# disco_gender,lpbs_mean,lpbs_std,SEAT_gender,tan_SEAT_gender, lau_SEAT_gender
# SEAT_EAvsAA, tan_SEAT_EAvsAA, lau_SEAT_EAvsAA,SEAT_EAvsAA,disco_black,disco_asian,disco_hispanic
# tan_SEAT_ABW,lau_SEAT_ABW ,SEAT_ABW
# 
# 
# 

extrinsic_metric = f'{chosen_data}_{names_included}_bias_{gender_or_race}_{average_or_median}_{bias_type}'
print(f'Constructed column name: {extrinsic_metric}')

# Check if the column exists in the DataFrame
if extrinsic_metric in df.columns:
    print(f'The column "{extrinsic_metric}" exists in the DataFrame.')
else:
    print(f'The column "{extrinsic_metric}" does not exist in the DataFrame.')
calculate_correlations(extrinsic_metric, intrinsic_metric)




"""
compute_correlation_matrix_for_subplot(
    intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
    extrinsic_columns=['o_wi_bias_f_mean_norm','o_wi_bias_f_median_norm','o_wi_bias_f_mean_norm','o_wi_bias_f_median_norm']
)
compute_correlation_matrix_for_subplot(
    intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
    extrinsic_columns=['o_w_bias_f_mean_norm','o_w_bias_f_median_norm','o_w_bias_f_mean_norm','o_w_bias_f_median_norm']
)
compute_correlation_matrix_for_subplot(
    intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
    extrinsic_columns=['r_wi_bias_f_mean_norm','r_wi_bias_f_median_norm','r_wi_bias_f_mean_norm','r_wi_bias_f_median_norm']
)
compute_correlation_matrix_for_subplot(
    intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
    extrinsic_columns=['r_w_bias_f_mean_norm','r_w_bias_f_median_norm','r_w_bias_f_mean_norm','r_w_bias_f_median_norm']
)
compute_correlation_matrix_for_subplot(
    intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
    extrinsic_columns=['p_wi_bias_f_mean_norm','p_wi_bias_f_median_norm','p_wi_bias_f_mean_norm','p_wi_bias_f_median_norm']
)
compute_correlation_matrix_for_subplot(
    intrinsic_columns=['disco_gender','lpbs_mean','SEAT_gender','tan_SEAT_gender', 'lau_SEAT_gender'],
    extrinsic_columns=['p_w_bias_f_mean_norm','p_w_bias_f_median_norm','p_w_bias_f_mean_norm','p_w_bias_f_median_norm']
)

compute_correlation_matrix_for_subplot(
    intrinsic_columns=['SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA','disco_black','disco_asian','disco_hispanic','tan_SEAT_ABW','lau_SEAT_ABW' ,'SEAT_ABW'],
    extrinsic_columns=['o_wi_bias_b_mean_norm','o_wi_bias_a_mean_norm','o_wi_bias_h_mean_norm','o_wi_bias_fb_mean_norm',    'o_wi_bias_b_median_norm','o_wi_bias_a_median_norm','o_wi_bias_h_median_norm','o_wi_bias_fb_median_norm',    'o_wi_bias_b_mean_norm','o_wi_bias_a_mean_norm','o_wi_bias_h_mean_norm','o_wi_bias_fb_mean_norm',    'o_wi_bias_b_median_norm','o_wi_bias_a_median_norm','o_wi_bias_h_median_norm','o_wi_bias_fb_median_norm']
)
compute_correlation_matrix_for_subplot(
    intrinsic_columns=['SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA','disco_black','disco_asian','disco_hispanic','tan_SEAT_ABW','lau_SEAT_ABW' ,'SEAT_ABW'],
    extrinsic_columns=['r_wi_bias_b_mean_norm','r_wi_bias_a_mean_norm','r_wi_bias_h_mean_norm','r_wi_bias_fb_mean_norm',    'r_wi_bias_b_median_norm','r_wi_bias_a_median_norm','r_wi_bias_h_median_norm','r_wi_bias_fb_median_norm',    'r_wi_bias_b_mean_norm','r_wi_bias_a_mean_norm','r_wi_bias_h_mean_norm','r_wi_bias_fb_mean_norm',    'r_wi_bias_b_median_norm','r_wi_bias_a_median_norm','r_wi_bias_h_median_norm','r_wi_bias_fb_median_norm']
)
compute_correlation_matrix_for_subplot(
    intrinsic_columns=['SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA','disco_black','disco_asian','disco_hispanic','tan_SEAT_ABW','lau_SEAT_ABW' ,'SEAT_ABW'],
    extrinsic_columns=['p_wi_bias_b_mean_norm','p_wi_bias_a_mean_norm','p_wi_bias_h_mean_norm','p_wi_bias_fb_mean_norm',    'p_wi_bias_b_median_norm','p_wi_bias_a_median_norm','p_wi_bias_h_median_norm','p_wi_bias_fb_median_norm',    'p_wi_bias_b_mean_norm','p_wi_bias_a_mean_norm','p_wi_bias_h_mean_norm','p_wi_bias_fb_mean_norm',    'p_wi_bias_b_median_norm','p_wi_bias_a_median_norm','p_wi_bias_h_median_norm','p_wi_bias_fb_median_norm']
)
compute_correlation_matrix_for_subplot(
    intrinsic_columns=['SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA','disco_black','disco_asian','disco_hispanic','tan_SEAT_ABW','lau_SEAT_ABW' ,'SEAT_ABW'],
    extrinsic_columns=['o_w_bias_b_mean_norm','o_w_bias_a_mean_norm','o_w_bias_h_mean_norm','o_w_bias_fb_mean_norm',    'o_w_bias_b_median_norm','o_w_bias_a_median_norm','o_w_bias_h_median_norm','o_w_bias_fb_median_norm',    'o_w_bias_b_mean_norm','o_w_bias_a_mean_norm','o_w_bias_h_mean_norm','o_w_bias_fb_mean_norm',    'o_w_bias_b_median_norm','o_w_bias_a_median_norm','o_w_bias_h_median_norm','o_w_bias_fb_median_norm']
)
compute_correlation_matrix_for_subplot(
    intrinsic_columns=['SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA','disco_black','disco_asian','disco_hispanic','tan_SEAT_ABW','lau_SEAT_ABW' ,'SEAT_ABW'],
    extrinsic_columns=['r_w_bias_b_mean_norm','r_w_bias_a_mean_norm','r_w_bias_h_mean_norm','r_w_bias_fb_mean_norm',    'r_w_bias_b_median_norm','r_w_bias_a_median_norm','r_w_bias_h_median_norm','r_w_bias_fb_median_norm',    'r_w_bias_b_mean_norm','r_w_bias_a_mean_norm','r_w_bias_h_mean_norm','r_w_bias_fb_mean_norm',    'r_w_bias_b_median_norm','r_w_bias_a_median_norm','r_w_bias_h_median_norm','r_w_bias_fb_median_norm']
)
compute_correlation_matrix_for_subplot(
    intrinsic_columns=['SEAT_EAvsAA', 'tan_SEAT_EAvsAA', 'lau_SEAT_EAvsAA','disco_black','disco_asian','disco_hispanic','tan_SEAT_ABW','lau_SEAT_ABW' ,'SEAT_ABW'],
    extrinsic_columns=['p_w_bias_b_mean_norm','p_w_bias_a_mean_norm','p_w_bias_h_mean_norm','p_w_bias_fb_mean_norm',    'p_w_bias_b_median_norm','p_w_bias_a_median_norm','p_w_bias_h_median_norm','p_w_bias_fb_median_norm',    'p_w_bias_b_mean_norm','p_w_bias_a_mean_norm','p_w_bias_h_mean_norm','p_w_bias_fb_mean_norm',    'p_w_bias_b_median_norm','p_w_bias_a_median_norm','p_w_bias_h_median_norm','o_wi_bias_fb_median_norm']
)"""



create_combined_figure_race()
create_combined_figure_gender()