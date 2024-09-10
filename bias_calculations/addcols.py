import pandas as pd
import numpy as np

# Load the existing CSV file
filename = 'intrinsic_bias.csv'
df = pd.read_csv(filename)

# Define the new columns to add
new_columns = [
    'o_wi_bias_a_mean_desc', 'o_w_bias_a_mean_desc', 'r_wi_bias_a_mean_desc', 'r_w_bias_a_mean_desc', 
    'b_wi_bias_a_mean_desc', 'b_w_bias_a_mean_desc', 'o_wi_bias_a_median_desc', 'o_w_bias_a_median_desc', 
    'r_wi_bias_a_median_desc', 'r_w_bias_a_median_desc', 'b_wi_bias_a_median_desc', 'b_w_bias_a_median_desc',
    'o_wi_bias_b_mean_desc', 'o_w_bias_b_mean_desc', 'r_wi_bias_b_mean_desc', 'r_w_bias_b_mean_desc',
    'b_wi_bias_b_mean_desc', 'b_w_bias_b_mean_desc', 'o_wi_bias_b_median_desc', 'o_w_bias_b_median_desc',
    'r_wi_bias_b_median_desc', 'r_w_bias_b_median_desc', 'b_wi_bias_b_median_desc', 'b_w_bias_b_median_desc',
    'o_wi_bias_h_mean_desc', 'o_w_bias_h_mean_desc', 'r_wi_bias_h_mean_desc', 'r_w_bias_h_mean_desc',
    'b_wi_bias_h_mean_desc', 'b_w_bias_h_mean_desc', 'o_wi_bias_h_median_desc', 'o_w_bias_h_median_desc',
    'r_wi_bias_h_median_desc', 'r_w_bias_h_median_desc', 'b_wi_bias_h_median_desc', 'b_w_bias_h_median_desc',
    'o_wi_bias_f_mean_desc', 'o_w_bias_f_mean_desc', 'r_wi_bias_f_mean_desc', 'r_w_bias_f_mean_desc',
    'b_wi_bias_f_mean_desc', 'b_w_bias_f_mean_desc', 'o_wi_bias_f_median_desc', 'o_w_bias_f_median_desc',
    'r_wi_bias_f_median_desc', 'r_w_bias_f_median_desc', 'b_wi_bias_f_median_desc', 'b_w_bias_f_median_desc',
    'o_wi_bias_fb_mean_desc', 'o_w_bias_fb_mean_desc', 'r_wi_bias_fb_mean_desc', 'r_w_bias_fb_mean_desc',
    'b_wi_bias_fb_mean_desc', 'b_w_bias_fb_mean_desc', 'o_wi_bias_fb_median_desc', 'o_w_bias_fb_median_desc',
    'r_wi_bias_fb_median_desc', 'r_w_bias_fb_median_desc', 'b_wi_bias_fb_median_desc', 'b_w_bias_fb_median_desc',
    'o_wi_bias_a_mean_norm', 'o_w_bias_a_mean_norm', 'r_wi_bias_a_mean_norm', 'r_w_bias_a_mean_norm',
    'b_wi_bias_a_mean_norm', 'b_w_bias_a_mean_norm', 'o_wi_bias_a_median_norm', 'o_w_bias_a_median_norm',
    'r_wi_bias_a_median_norm', 'r_w_bias_a_median_norm', 'b_wi_bias_a_median_norm', 'b_w_bias_a_median_norm',
    'o_wi_bias_b_mean_norm', 'o_w_bias_b_mean_norm', 'r_wi_bias_b_mean_norm', 'r_w_bias_b_mean_norm',
    'b_wi_bias_b_mean_norm', 'b_w_bias_b_mean_norm', 'o_wi_bias_b_median_norm', 'o_w_bias_b_median_norm',
    'r_wi_bias_b_median_norm', 'r_w_bias_b_median_norm', 'b_wi_bias_b_median_norm', 'b_w_bias_b_median_norm',
    'o_wi_bias_h_mean_norm', 'o_w_bias_h_mean_norm', 'r_wi_bias_h_mean_norm', 'r_w_bias_h_mean_norm',
    'b_wi_bias_h_mean_norm', 'b_w_bias_h_mean_norm', 'o_wi_bias_h_median_norm', 'o_w_bias_h_median_norm',
    'r_wi_bias_h_median_norm', 'r_w_bias_h_median_norm', 'b_wi_bias_h_median_norm', 'b_w_bias_h_median_norm',
    'o_wi_bias_f_mean_norm', 'o_w_bias_f_mean_norm', 'r_wi_bias_f_mean_norm', 'r_w_bias_f_mean_norm',
    'b_wi_bias_f_mean_norm', 'b_w_bias_f_mean_norm', 'o_wi_bias_f_median_norm', 'o_w_bias_f_median_norm',
    'r_wi_bias_f_median_norm', 'r_w_bias_f_median_norm', 'b_wi_bias_f_median_norm', 'b_w_bias_f_median_norm',
    'o_wi_bias_fb_mean_norm', 'o_w_bias_fb_mean_norm', 'r_wi_bias_fb_mean_norm', 'r_w_bias_fb_mean_norm',
    'b_wi_bias_fb_mean_norm', 'b_w_bias_fb_mean_norm', 'o_wi_bias_fb_median_norm', 'o_w_bias_fb_median_norm',
    'r_wi_bias_fb_median_norm', 'r_w_bias_fb_median_norm', 'b_wi_bias_fb_median_norm', 'b_w_bias_fb_median_norm'
]

# Add new columns to DataFrame with NaN values
for column in new_columns:
    df[column] = np.nan  # You can replace np.nan with a different placeholder if needed

# Save the updated DataFrame back to CSV
df.to_csv(filename, index=False)
