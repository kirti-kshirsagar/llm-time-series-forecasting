"""
Data Cleaning and Preprocessing Module
====================================

This module handles data quality issues and prepares the data for LLM adaptation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """
    Handles data cleaning and preprocessing for the volume forecasting task.
    """
    
    def __init__(self):
        self.opened_data = None
        self.closed_data = None
        self.cleaned_opened = None
        self.cleaned_closed = None
        self.merged_data = None
        self.cleaning_log = []
        
    def load_raw_data(self):
        """Load the raw datasets."""
        print("Loading raw datasets...")
        self.opened_data = pd.read_csv("../data/requests_opened_external.csv")
        self.closed_data = pd.read_csv("../data/requests_closed_external.csv")
        print(f"✓ Loaded opened data: {self.opened_data.shape}")
        print(f"✓ Loaded closed data: {self.closed_data.shape}")
        
    def clean_opened_data(self):
        """
        Clean and preprocess the opened dataset.
        Key issues to address:
        1. Convert Date/Time to proper datetime
        2. Aggregate minute-level data to daily
        3. Handle outliers
        4. Align with assignment description (tickets_received)
        """
        print("\n" + "=" * 60)
        print("CLEANING OPENED DATASET")
        print("=" * 60)
        
        # Create a copy for cleaning
        df = self.opened_data.copy()
        
        # Step 1: Create proper datetime column
        print("Step 1: Creating datetime column...")
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M')
        print(f"✓ Created datetime column")
        
        # Step 2: Aggregate to daily level
        print("Step 2: Aggregating to daily level...")
        daily_aggregated = df.groupby(df['datetime'].dt.date).agg({
            'Volume': 'sum'  # Sum all volumes for the day
        }).reset_index()
        
        # Rename columns to match description
        daily_aggregated.columns = ['date', 'tickets_received']
        daily_aggregated['date'] = pd.to_datetime(daily_aggregated['date'])
        
        print(f"✓ Aggregated from {len(df)} records to {len(daily_aggregated)} daily records")
        
        # Step 3: Handle outliers using IQR method
        print("Step 3: Handling outliers...")
        Q1 = daily_aggregated['tickets_received'].quantile(0.25)
        Q3 = daily_aggregated['tickets_received'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (daily_aggregated['tickets_received'] < lower_bound) | \
                       (daily_aggregated['tickets_received'] > upper_bound)
        outliers_count = outliers_mask.sum()
        
        print(f"✓ Identified {outliers_count} outliers ({outliers_count/len(daily_aggregated)*100:.1f}%)")
        print(f"  Outlier bounds: [{lower_bound:,.0f}, {upper_bound:,.0f}]")
        
        # Remove outliers
        daily_clean = daily_aggregated[~outliers_mask].copy()
        print(f"✓ Removed outliers, final shape: {daily_clean.shape}")
        
        # Step 4: Sort by date
        daily_clean = daily_clean.sort_values('date').reset_index(drop=True)
        
        # Store cleaned data
        self.cleaned_opened = daily_clean
        
        # Log cleaning steps
        self.cleaning_log.append({
            'step': 'opened_cleaning',
            'original_shape': self.opened_data.shape,
            'final_shape': daily_clean.shape,
            'outliers_removed': outliers_count,
            'date_range': [str(daily_clean['date'].min()), str(daily_clean['date'].max())]
        })
        
        return daily_clean
        
    def clean_closed_data(self):
        """
        Clean and preprocess the closed dataset.
        Key issues to address:
        1. Convert date to proper datetime
        2. Handle missing dates (weekends/holidays)
        3. Align with description (tickets_resolved)
        """
        print("\n" + "=" * 60)
        print("CLEANING CLOSED DATASET")
        print("=" * 60)
        
        # Create a copy for cleaning
        df = self.closed_data.copy()
        
        # Step 1: Convert date to datetime
        print("Step 1: Converting date to datetime...")
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Converted date column")
        
        # Step 2: Rename volume to tickets_resolved
        print("Step 2: Renaming columns...")
        df = df.rename(columns={'volume': 'tickets_resolved'})
        print(f"✓ Renamed volume to tickets_resolved")
        
        # Step 3: Handle missing dates
        print("Step 3: Handling missing dates...")
        date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
        missing_dates = set(date_range.date) - set(df['date'].dt.date)
        print(f"✓ Found {len(missing_dates)} missing dates")
        
        # Create complete date range and fill missing values
        complete_df = pd.DataFrame({'date': date_range})
        complete_df = complete_df.merge(df, on='date', how='left')
        
        # Fill missing values with 0 (assuming no tickets resolved on weekends/holidays)
        complete_df['tickets_resolved'] = complete_df['tickets_resolved'].fillna(0)
        
        print(f"✓ Filled missing dates with 0, final shape: {complete_df.shape}")
        
        # Store cleaned data
        self.cleaned_closed = complete_df
        
        # Log cleaning steps
        self.cleaning_log.append({
            'step': 'closed_cleaning',
            'original_shape': self.closed_data.shape,
            'final_shape': complete_df.shape,
            'missing_dates_filled': len(missing_dates),
            'date_range': [str(complete_df['date'].min()), str(complete_df['date'].max())]
        })
        
        return complete_df
        
    def merge_datasets(self):
        """
        Merge the cleaned opened and closed datasets.
        """
        print("\n" + "=" * 60)
        print("MERGING DATASETS")
        print("=" * 60)
        
        # Merge on date
        merged = pd.merge(
            self.cleaned_opened, 
            self.cleaned_closed, 
            on='date', 
            how='outer'
        )
        
        # Sort by date
        merged = merged.sort_values('date').reset_index(drop=True)
        
        # Fill any remaining missing values
        merged['tickets_received'] = merged['tickets_received'].fillna(0)
        merged['tickets_resolved'] = merged['tickets_resolved'].fillna(0)
        
        print(f"✓ Merged datasets, final shape: {merged.shape}")
        print(f"✓ Date range: {merged['date'].min()} to {merged['date'].max()}")
        print(f"✓ Total days: {len(merged)}")
        
        # Store merged data
        self.merged_data = merged
        
        # Log merging
        self.cleaning_log.append({
            'step': 'merging',
            'opened_shape': self.cleaned_opened.shape,
            'closed_shape': self.cleaned_closed.shape,
            'merged_shape': merged.shape,
            'date_range': [str(merged['date'].min()), str(merged['date'].max())]
        })
        
        return merged
        
    def create_visualizations(self):
        """Create visualizations of the cleaned data."""
        print("\n" + "=" * 60)
        print("CREATING CLEANED DATA VISUALIZATIONS")
        print("=" * 60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cleaned Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Time series of tickets received
        sample_data = self.merged_data.head(1000)  # Sample for visualization
        axes[0, 0].plot(sample_data['date'], sample_data['tickets_received'])
        axes[0, 0].set_title('Tickets Received Over Time (First 1000 days)')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Tickets Received')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Time series of tickets resolved
        axes[0, 1].plot(sample_data['date'], sample_data['tickets_resolved'])
        axes[0, 1].set_title('Tickets Resolved Over Time (First 1000 days)')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Tickets Resolved')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Distribution of tickets received
        axes[1, 0].hist(self.merged_data['tickets_received'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribution of Daily Tickets Received')
        axes[1, 0].set_xlabel('Tickets Received')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Correlation between received and resolved
        axes[1, 1].scatter(self.merged_data['tickets_received'], self.merged_data['tickets_resolved'], alpha=0.5)
        axes[1, 1].set_title('Tickets Received vs Resolved')
        axes[1, 1].set_xlabel('Tickets Received')
        axes[1, 1].set_ylabel('Tickets Resolved')
        
        # Add correlation coefficient
        corr = self.merged_data['tickets_received'].corr(self.merged_data['tickets_resolved'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('../results/cleaned_data_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizations saved as 'cleaned_data_visualizations.png'")
        
    def generate_summary_report(self):
        """Generate a summary report of the cleaning process."""
        print("\n" + "=" * 60)
        print("CLEANING SUMMARY REPORT")
        print("=" * 60)
        
        print("📊 DATASET OVERVIEW:")
        print(f"  • Opened data: {self.opened_data.shape[0]:,} records → {self.cleaned_opened.shape[0]:,} daily records")
        print(f"  • Closed data: {self.closed_data.shape[0]:,} records → {self.cleaned_closed.shape[0]:,} daily records")
        print(f"  • Merged data: {self.merged_data.shape[0]:,} daily records")
        
        print("\n📅 DATE RANGES:")
        print(f"  • Opened: {self.cleaned_opened['date'].min()} to {self.cleaned_opened['date'].max()}")
        print(f"  • Closed: {self.cleaned_closed['date'].min()} to {self.cleaned_closed['date'].max()}")
        print(f"  • Merged: {self.merged_data['date'].min()} to {self.merged_data['date'].max()}")
        
        print("\n📈 DATA STATISTICS:")
        print(f"  • Tickets received - Mean: {self.merged_data['tickets_received'].mean():,.0f}, Std: {self.merged_data['tickets_received'].std():,.0f}")
        print(f"  • Tickets resolved - Mean: {self.merged_data['tickets_resolved'].mean():,.0f}, Std: {self.merged_data['tickets_resolved'].std():,.0f}")
        
        print("\n🔧 CLEANING ACTIONS TAKEN:")
        for log_entry in self.cleaning_log:
            print(f"  • {log_entry['step']}: {log_entry}")
        
        # Save summary to file
        import json
        with open('../results/cleaning_summary.json', 'w') as f:
            json.dump(self.cleaning_log, f, indent=2, default=str)
        
        print("\n✓ Summary saved to 'cleaning_summary.json'")
        
    def run_complete_cleaning(self):
        """Run the complete data cleaning pipeline."""
        print("=" * 80)
        print("STARTING DATA CLEANING AND PREPROCESSING")
        print("=" * 80)
        
        # Load data
        self.load_raw_data()
        
        # Clean datasets
        self.clean_opened_data()
        self.clean_closed_data()
        
        # Merge datasets
        self.merge_datasets()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate summary
        self.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("DATA CLEANING COMPLETE")
        print("=" * 80)
        print("✓ All datasets cleaned and merged")
        print("✓ Visualizations created")
        print("✓ Summary report generated")
        print("✓ Ready for feature engineering and LLM adaptation")
        
        return self.merged_data

if __name__ == "__main__":
    # Run complete cleaning pipeline
    cleaner = DataCleaner()
    cleaned_data = cleaner.run_complete_cleaning()
    
    # Save cleaned data
    cleaned_data.to_csv('../data/cleaned_merged_data.csv', index=False)
    print("✓ Cleaned data saved to 'cleaned_merged_data.csv'")
