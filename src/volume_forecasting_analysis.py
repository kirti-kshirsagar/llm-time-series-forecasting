"""
Take Home Assignment: ML Researcher - Volume Forecasting
=======================================================

Objective: Transform an LLM into a time series forecasting model to predict 
daily customer support ticket volumes.

Author: Kirti
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VolumeForecastingAnalysis:
    """
    Main class for the volume forecasting analysis and LLM adaptation.
    """
    
    def __init__(self):
        self.opened_data = None
        self.closed_data = None
        self.processed_data = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load and perform initial inspection of both datasets."""
        print("=" * 60)
        print("STEP 1: DATA LOADING AND INITIAL INSPECTION")
        print("=" * 60)
        
        # Load datasets
        print("Loading datasets...")
        self.opened_data = pd.read_csv("../data/requests_opened_external.csv")
        self.closed_data = pd.read_csv("../data/requests_closed_external.csv")
        
        print(f"✓ Opened dataset loaded: {self.opened_data.shape}")
        print(f"✓ Closed dataset loaded: {self.closed_data.shape}")
        
        # Initial inspection
        self._inspect_datasets()
        
    def _inspect_datasets(self):
        """Perform detailed inspection of both datasets."""
        print("\n" + "-" * 40)
        print("DETAILED DATASET INSPECTION")
        print("-" * 40)
        
        # Opened dataset inspection
        print("\n📊 OPENED DATASET ANALYSIS:")
        print(f"Shape: {self.opened_data.shape}")
        print(f"Columns: {list(self.opened_data.columns)}")
        print(f"Data types:\n{self.opened_data.dtypes}")
        print(f"Missing values:\n{self.opened_data.isnull().sum()}")
        print(f"Sample data:\n{self.opened_data.head()}")
        
        # Check for data quality issues
        print(f"\nDate column unique values (first 10): {self.opened_data['Date'].unique()[:10]}")
        print(f"Time column unique values (first 10): {self.opened_data['Time'].unique()[:10]}")
        print(f"Volume statistics:\n{self.opened_data['Volume'].describe()}")
        
        # Closed dataset inspection
        print("\n📊 CLOSED DATASET ANALYSIS:")
        print(f"Shape: {self.closed_data.shape}")
        print(f"Columns: {list(self.closed_data.columns)}")
        print(f"Data types:\n{self.closed_data.dtypes}")
        print(f"Missing values:\n{self.closed_data.isnull().sum()}")
        print(f"Sample data:\n{self.closed_data.head()}")
        
        print(f"\nDate range: {self.closed_data['date'].min()} to {self.closed_data['date'].max()}")
        print(f"Volume statistics:\n{self.closed_data['volume'].describe()}")
        
        # Store analysis results
        self.analysis_results['opened_shape'] = self.opened_data.shape
        self.analysis_results['closed_shape'] = self.closed_data.shape
        self.analysis_results['opened_missing'] = self.opened_data.isnull().sum().to_dict()
        self.analysis_results['closed_missing'] = self.closed_data.isnull().sum().to_dict()
        
    def explore_data_quality(self):
        """Explore data quality issues in detail."""
        print("\n" + "=" * 60)
        print("STEP 2: DATA QUALITY ASSESSMENT")
        print("=" * 60)
        
        # Analyze opened dataset quality issues
        print("\n🔍 OPENED DATASET QUALITY ANALYSIS:")
        
        # Date format analysis
        date_samples = self.opened_data['Date'].head(20)
        print(f"Date format samples: {date_samples.tolist()}")
        
        # Time format analysis
        time_samples = self.opened_data['Time'].head(20)
        print(f"Time format samples: {time_samples.tolist()}")
        
        # Volume analysis
        volume_stats = self.opened_data['Volume'].describe()
        print(f"Volume statistics:\n{volume_stats}")
        
        # Check for anomalies in volume
        volume_q99 = self.opened_data['Volume'].quantile(0.99)
        volume_q01 = self.opened_data['Volume'].quantile(0.01)
        print(f"Volume 99th percentile: {volume_q99:,.0f}")
        print(f"Volume 1st percentile: {volume_q01:,.0f}")
        
        # Check for zero or negative volumes
        zero_volumes = (self.opened_data['Volume'] <= 0).sum()
        print(f"Zero or negative volumes: {zero_volumes}")
        
        # Analyze closed dataset
        print("\n🔍 CLOSED DATASET QUALITY ANALYSIS:")
        closed_date_range = pd.to_datetime(self.closed_data['date'])
        print(f"Date range: {closed_date_range.min()} to {closed_date_range.max()}")
        print(f"Total days: {(closed_date_range.max() - closed_date_range.min()).days}")
        
        # Check for gaps in closed data
        closed_dates = pd.to_datetime(self.closed_data['date']).dt.date
        expected_dates = pd.date_range(closed_date_range.min().date(), closed_date_range.max().date(), freq='D')
        missing_dates = set(expected_dates.date) - set(closed_dates)
        print(f"Missing dates in closed data: {len(missing_dates)}")
        
        # Store quality assessment
        self.analysis_results['quality_issues'] = {
            'opened_zero_volumes': zero_volumes,
            'opened_volume_range': [volume_q01, volume_q99],
            'closed_missing_dates': len(missing_dates),
            'closed_date_range': [str(closed_date_range.min()), str(closed_date_range.max())]
        }
        
    def visualize_data(self):
        """Create comprehensive visualizations of the data."""
        print("\n" + "=" * 60)
        print("STEP 3: DATA VISUALIZATION")
        print("=" * 60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Exploration Visualizations', fontsize=16, fontweight='bold')
        
        # 1. Volume distribution in opened data
        axes[0, 0].hist(self.opened_data['Volume'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution of Volume in Opened Data')
        axes[0, 0].set_xlabel('Volume')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_yscale('log')
        
        # 2. Volume distribution in closed data
        axes[0, 1].hist(self.closed_data['volume'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution of Volume in Closed Data')
        axes[0, 1].set_xlabel('Volume')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        
        # 3. Time series plot for closed data (sample)
        closed_sample = self.closed_data.head(1000)  # Sample for visualization
        closed_sample['date_parsed'] = pd.to_datetime(closed_sample['date'])
        axes[1, 0].plot(closed_sample['date_parsed'], closed_sample['volume'])
        axes[1, 0].set_title('Time Series: Closed Data (First 1000 points)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Volume')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Box plot comparison
        data_for_box = [self.opened_data['Volume'], self.closed_data['volume']]
        axes[1, 1].boxplot(data_for_box, labels=['Opened', 'Closed'])
        axes[1, 1].set_title('Volume Distribution Comparison')
        axes[1, 1].set_ylabel('Volume (log scale)')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('../results/data_exploration_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizations saved as 'data_exploration_visualizations.png'")
        
    def run_initial_analysis(self):
        """Run the complete initial analysis."""
        self.load_data()
        self.explore_data_quality()
        self.visualize_data()
        
        # Save analysis results
        with open('../results/initial_analysis_results.json', 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print("\n" + "=" * 60)
        print("INITIAL ANALYSIS COMPLETE")
        print("=" * 60)
        print("✓ Data loaded and inspected")
        print("✓ Quality issues identified")
        print("✓ Visualizations created")
        print("✓ Results saved to 'initial_analysis_results.json'")
        
        return self.analysis_results

if __name__ == "__main__":
    # Initialize and run analysis
    analysis = VolumeForecastingAnalysis()
    results = analysis.run_initial_analysis()
    
    print("\nNext steps:")
    print("1. Review the analysis results")
    print("2. Proceed with data cleaning and preprocessing")
    print("3. Design LLM adaptation strategy")
