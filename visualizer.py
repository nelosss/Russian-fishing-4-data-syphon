import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import glob
import re
import warnings
import numpy as np

# Suppress specific FutureWarning from seaborn regarding palette usage without hue
warnings.simplefilter(action='ignore', category=FutureWarning)


def clean_weight(weight_str):
    """Extracts numeric weight in grams from 'Weight/Size' string."""
    if pd.isna(weight_str):
        return None
    cleaned = re.sub(r'[^\d\s]', '', str(weight_str)).strip()
    cleaned = cleaned.replace(' ', '')
    if cleaned.isdigit():
        weight = int(cleaned)
        return weight if weight > 0 else None
    return None

# Removed get_time_details function

# generate_region_plots function is removed for now to focus on combined plot

def generate_combined_plots(df_combined, base_dir):
    """Generates plots based on the combined data from all regions."""
    print("\n--- Generating plots for COMBINED data ---")
    output_plot_dir = os.path.join(base_dir, 'combined_plots')
    os.makedirs(output_plot_dir, exist_ok=True)
    print(f"  Saving plots to: {output_plot_dir}")

    if df_combined.empty:
        print("  Skipping combined plots due to empty DataFrame.")
        return

    sns.set_theme(style="whitegrid")
    # Define Top N limits for the scatter plot
    top_n_fish_scatter = 15
    top_n_baits_scatter = 20

    # --- Combined Data Plot: Fish vs Bait Scatter (Size=Count, Color=Avg Weight) ---
    if 'Fish Name' in df_combined.columns and 'Bait/Tackle' in df_combined.columns and 'Weight (g)' in df_combined.columns:
        print("  Generating Combined Bait vs Fish Scatter Plot (Size=Count, Color=Avg Weight)...")
        try:
            # Get overall top fish and baits
            top_fish_scatter_names = df_combined['Fish Name'].value_counts().nlargest(top_n_fish_scatter).index.tolist()
            baits_exploded_scatter = df_combined['Bait/Tackle'].dropna().astype(str).str.split(',').explode()
            baits_exploded_scatter = baits_exploded_scatter.str.strip()
            baits_exploded_scatter = baits_exploded_scatter[baits_exploded_scatter != '']
            top_baits_scatter_names = baits_exploded_scatter.value_counts().nlargest(top_n_baits_scatter).index.tolist()

            # Filter dataframe to include only top fish/baits and explode baits again
            df_scatter_filtered = df_combined[df_combined['Fish Name'].isin(top_fish_scatter_names)].copy()
            df_scatter_filtered['Bait'] = df_scatter_filtered['Bait/Tackle'].dropna().astype(str).str.split(',')
            df_scatter_exploded = df_scatter_filtered.explode('Bait')
            df_scatter_exploded['Bait'] = df_scatter_exploded['Bait'].str.strip()
            df_scatter_final = df_scatter_exploded[df_scatter_exploded['Bait'].isin(top_baits_scatter_names)]

            if not df_scatter_final.empty:
                # Calculate counts AND average weight for each combination
                scatter_data = df_scatter_final.groupby(['Fish Name', 'Bait']).agg(
                    Count=('Weight (g)', 'size'),
                    AvgWeight=('Weight (g)', 'mean')
                ).reset_index()

                if not scatter_data.empty:
                    # Determine appropriate size scaling
                    min_size, max_size = 30, 600 # Adjust min/max dot size
                    size_range = (min_size, max_size)

                    plt.figure(figsize=(15, 10)) # Adjust size
                    scatter_plot = sns.scatterplot(
                        data=scatter_data,
                        x='Bait',
                        y='Fish Name',
                        size='Count',
                        sizes=size_range,
                        hue='AvgWeight', # Color by average weight
                        palette='coolwarm', # Use a diverging palette for weight
                        legend=False, # Turn off default legend, we'll make custom ones
                        alpha=0.8
                    )
                    plt.title(f'Top {top_n_fish_scatter} Fish vs Top {top_n_baits_scatter} Baits (Size=Count, Color=Avg Weight) - All Regions')
                    plt.xlabel('Bait/Tackle')
                    plt.ylabel('Fish Species')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)

                    # --- Simplified Legend Creation ---
                    # Create size legend manually
                    size_legend_handles = [plt.scatter([],[], s=s, color='gray', alpha=0.7) for s in np.linspace(min_size, max_size, 5)]
                    size_legend_labels = [f"{int(c)}" for c in np.linspace(scatter_data['Count'].min(), scatter_data['Count'].max(), 5)]
                    size_leg = plt.legend(size_legend_handles, size_legend_labels, title='Count', bbox_to_anchor=(1.15, 0.5), loc='center left', frameon=False, labelspacing=1.5)
                    plt.gca().add_artist(size_leg) # Add size legend manually

                    # Create hue legend (colorbar)
                    norm = plt.Normalize(scatter_data['AvgWeight'].min(), scatter_data['AvgWeight'].max())
                    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
                    sm.set_array([])
                    # Position colorbar relative to the axes
                    cbar = plt.colorbar(sm, ax=plt.gca(), anchor=(1.0, 0.5), shrink=0.6, pad=0.15) # Adjust pad
                    cbar.set_label('Average Weight (g)')
                    # --- End Simplified Legend Creation ---

                    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to prevent legend overlap
                    plot_file = os.path.join(output_plot_dir, 'combined_scatter_fish_bait_size_color.png')
                    plt.savefig(plot_file)
                    print(f"  Saved: {os.path.basename(plot_file)}")
                    plt.close()
                else:
                    print("  Skipping: Scatter Plot (no data after filtering/grouping)")
            else:
                 print("  Skipping: Scatter Plot (no data for top fish/bait combinations)")
        except Exception as e_scatter:
            print(f"  Skipping: Scatter Plot (Error processing: {e_scatter})")

    print("--- Finished combined plots ---")


def main():
    base_dir = os.getcwd()
    all_subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d != 'output_plots' and d != 'combined_plots']

    if not all_subdirs:
        print("No region subdirectories found.")
        return

    print(f"Found {len(all_subdirs)} potential region directories. Loading all data...")

    all_df_list = []
    processed_regions = 0
    for region_dir in all_subdirs:
        region_csv_path = os.path.join(base_dir, region_dir, 'data.csv')
        if os.path.exists(region_csv_path):
            # print(f"  Loading data from: {region_dir}") # Reduce noise
            try:
                df_region = pd.read_csv(region_csv_path, dtype={
                    'Timestamp?': str, 'Map/Region': str, 'Category?': str,
                    'Fish Name': str, 'Weight/Size': str, 'Bait/Tackle': str,
                    'Trophy': str, 'Blue Tag': str, 'Rare': str, 'Super Rare': str,
                    'Country Code': str
                })
                if df_region.empty:
                    # print(f"    Skipping empty file: {region_csv_path}")
                    continue

                df_region['Map/Region'] = region_dir
                all_df_list.append(df_region)
                processed_regions += 1
            except Exception as e:
                print(f"    Error loading file {region_csv_path}: {e}")
        else:
            pass

    if not all_df_list:
         print("\nNo data loaded from any region files. Cannot generate plots.")
         return

    df_combined = pd.concat(all_df_list, ignore_index=True)
    print(f"\nCombined data from {processed_regions} regions into a single DataFrame with {len(df_combined)} rows.")

    # --- Clean Combined Data ---
    print("  Cleaning combined data...")
    if 'Weight/Size' in df_combined.columns:
        df_combined['Weight (g)'] = df_combined['Weight/Size'].apply(clean_weight)
        df_combined.dropna(subset=['Weight (g)'], inplace=True)
        if not df_combined.empty:
            df_combined['Weight (g)'] = df_combined['Weight (g)'].astype(int)
    df_combined['Is Trophy'] = df_combined['Trophy'].apply(lambda x: 1 if x == 'Yes' else 0) if 'Trophy' in df_combined.columns else 0
    df_combined['Is Rare'] = df_combined['Rare'].apply(lambda x: 1 if x == 'Yes' else 0) if 'Rare' in df_combined.columns else 0
    df_combined['Is Super Rare'] = df_combined['Super Rare'].apply(lambda x: 1 if x == 'Yes' else 0) if 'Super Rare' in df_combined.columns else 0
    def get_category(row):
        if row['Is Super Rare'] == 1: return 'Super Rare'
        if row['Is Rare'] == 1: return 'Rare'
        if row['Is Trophy'] == 1: return 'Trophy'
        return 'Common'
    df_combined['Catch Category'] = df_combined.apply(get_category, axis=1)
    print(f"  Proceeding with {len(df_combined)} valid rows in combined data.")

    # --- Generate Combined Plots ONLY ---
    if not df_combined.empty:
        generate_combined_plots(df_combined, base_dir)
    else:
        print("\nSkipping combined plots as DataFrame is empty after cleaning.")

    # --- Skip Per-Region Plots for now ---
    # if processed_regions > 0 and not df_combined.empty:
    #      print("\n--- Generating Per-Region Plots ---")
    #      # Group combined data by region and process each group
    #      for region_name, df_region_group in df_combined.groupby('Map/Region'):
    #           generate_region_plots(df_region_group, region_name, base_dir)
    #      print(f"\nFinished generating plots for {processed_regions} regions.")
    # elif processed_regions > 0:
    #      print("\nSkipping per-region plots as combined DataFrame is empty after cleaning.")


if __name__ == "__main__":
    main()
