"""
Feature Engineering Module
=========================

This module creates temporal features, lag features, and prepares data for LLM adaptation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Handles feature engineering for time series forecasting.
    """
    
    def __init__(self):
        self.data = None
        self.features_data = None
        self.feature_names = []
        self.feature_importance = {}
        
    def load_cleaned_data(self):
        """Load the cleaned and merged data."""
        print("Loading cleaned data...")
        self.data = pd.read_csv('../data/cleaned_merged_data.csv')
        self.data['date'] = pd.to_datetime(self.data['date'])
        print(f"✓ Loaded data: {self.data.shape}")
        print(f"✓ Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        
    def create_temporal_features(self):
        """Create temporal features from the date column."""
        print("\n" + "=" * 60)
        print("CREATING TEMPORAL FEATURES")
        print("=" * 60)
        
        df = self.data.copy()
        
        # Basic temporal features
        print("Creating basic temporal features...")
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # Cyclical encoding for temporal features
        print("Creating cyclical encodings...")
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Business day features
        print("Creating business day features...")
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Holiday-like features (simplified)
        print("Creating holiday-like features...")
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        # Store feature names
        temporal_features = [
            'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            'day_of_year_sin', 'day_of_year_cos', 'is_weekend', 'is_monday', 'is_friday',
            'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end'
        ]
        self.feature_names.extend(temporal_features)
        
        print(f"✓ Created {len(temporal_features)} temporal features")
        
        self.data = df
        return df
        
    def create_lag_features(self, max_lags=14):
        """Create lag features for the target variable."""
        print(f"\n" + "=" * 60)
        print(f"CREATING LAG FEATURES (max_lags={max_lags})")
        print("=" * 60)
        
        df = self.data.copy()
        
        # Create lag features for tickets_received
        print("Creating lag features for tickets_received...")
        for lag in range(1, max_lags + 1):
            df[f'tickets_received_lag_{lag}'] = df['tickets_received'].shift(lag)
            self.feature_names.append(f'tickets_received_lag_{lag}')
        
        # Create lag features for tickets_resolved
        print("Creating lag features for tickets_resolved...")
        for lag in range(1, max_lags + 1):
            df[f'tickets_resolved_lag_{lag}'] = df['tickets_resolved'].shift(lag)
            self.feature_names.append(f'tickets_resolved_lag_{lag}')
        
        print(f"✓ Created {max_lags * 2} lag features")
        
        self.data = df
        return df
        
    def create_rolling_features(self, windows=[3, 7, 14, 30]):
        """Create rolling window features."""
        print(f"\n" + "=" * 60)
        print(f"CREATING ROLLING WINDOW FEATURES (windows={windows})")
        print("=" * 60)
        
        df = self.data.copy()
        
        for window in windows:
            print(f"Creating {window}-day rolling features...")
            
            # Rolling statistics for tickets_received
            df[f'tickets_received_rolling_mean_{window}'] = df['tickets_received'].rolling(window=window).mean()
            df[f'tickets_received_rolling_std_{window}'] = df['tickets_received'].rolling(window=window).std()
            df[f'tickets_received_rolling_min_{window}'] = df['tickets_received'].rolling(window=window).min()
            df[f'tickets_received_rolling_max_{window}'] = df['tickets_received'].rolling(window=window).max()
            df[f'tickets_received_rolling_median_{window}'] = df['tickets_received'].rolling(window=window).median()
            
            # Rolling statistics for tickets_resolved
            df[f'tickets_resolved_rolling_mean_{window}'] = df['tickets_resolved'].rolling(window=window).mean()
            df[f'tickets_resolved_rolling_std_{window}'] = df['tickets_resolved'].rolling(window=window).std()
            
            # Add to feature names
            rolling_features = [
                f'tickets_received_rolling_mean_{window}',
                f'tickets_received_rolling_std_{window}',
                f'tickets_received_rolling_min_{window}',
                f'tickets_received_rolling_max_{window}',
                f'tickets_received_rolling_median_{window}',
                f'tickets_resolved_rolling_mean_{window}',
                f'tickets_resolved_rolling_std_{window}'
            ]
            self.feature_names.extend(rolling_features)
        
        print(f"✓ Created {len(windows) * 7} rolling window features")
        
        self.data = df
        return df
        
    def create_difference_features(self):
        """Create difference features (day-to-day changes)."""
        print(f"\n" + "=" * 60)
        print("CREATING DIFFERENCE FEATURES")
        print("=" * 60)
        
        df = self.data.copy()
        
        # First differences
        df['tickets_received_diff_1'] = df['tickets_received'].diff(1)
        df['tickets_resolved_diff_1'] = df['tickets_resolved'].diff(1)
        
        # Second differences
        df['tickets_received_diff_2'] = df['tickets_received'].diff(2)
        df['tickets_resolved_diff_2'] = df['tickets_resolved'].diff(2)
        
        # Weekly differences
        df['tickets_received_diff_7'] = df['tickets_received'].diff(7)
        df['tickets_resolved_diff_7'] = df['tickets_resolved'].diff(7)
        
        # Add to feature names
        diff_features = [
            'tickets_received_diff_1', 'tickets_resolved_diff_1',
            'tickets_received_diff_2', 'tickets_resolved_diff_2',
            'tickets_received_diff_7', 'tickets_resolved_diff_7'
        ]
        self.feature_names.extend(diff_features)
        
        print(f"✓ Created {len(diff_features)} difference features")
        
        self.data = df
        return df
        
    def create_ratio_features(self):
        """Create ratio features between received and resolved tickets."""
        print(f"\n" + "=" * 60)
        print("CREATING RATIO FEATURES")
        print("=" * 60)
        
        df = self.data.copy()
        
        # Basic ratios
        df['resolved_to_received_ratio'] = df['tickets_resolved'] / (df['tickets_received'] + 1)  # +1 to avoid division by zero
        df['received_to_resolved_ratio'] = df['tickets_received'] / (df['tickets_resolved'] + 1)
        
        # Backlog estimation (simplified)
        df['estimated_backlog'] = df['tickets_received'] - df['tickets_resolved']
        
        # Add to feature names
        ratio_features = [
            'resolved_to_received_ratio', 'received_to_resolved_ratio', 'estimated_backlog'
        ]
        self.feature_names.extend(ratio_features)
        
        print(f"✓ Created {len(ratio_features)} ratio features")
        
        self.data = df
        return df
        
    def create_llm_text_features(self):
        """
        Create text-based features for LLM adaptation.
        This is a key innovation for adapting LLMs to time series forecasting.
        """
        print(f"\n" + "=" * 60)
        print("CREATING LLM TEXT FEATURES")
        print("=" * 60)
        
        df = self.data.copy()
        
        # Create text representations of temporal patterns
        print("Creating text representations...")
        
        # Day of week text
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_name'] = df['day_of_week'].map(lambda x: day_names[x])
        
        # Month name
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df['month_name'] = df['month'].map(lambda x: month_names[x-1])
        
        # Season
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df['season'] = df['month'].apply(get_season)
        
        # Create descriptive text for each day
        print("Creating descriptive text patterns...")
        df['temporal_description'] = (
            df['day_name'] + '_' + df['month_name'] + '_' + 
            df['season'] + '_' + df['year'].astype(str)
        )
        
        # Create volume level descriptions
        def get_volume_level(volume):
            if volume < df['tickets_received'].quantile(0.25):
                return 'low'
            elif volume < df['tickets_received'].quantile(0.75):
                return 'medium'
            else:
                return 'high'
        
        df['volume_level'] = df['tickets_received'].apply(get_volume_level)
        
        # Create comprehensive text feature for LLM
        df['llm_text_feature'] = (
            "Date: " + df['date'].dt.strftime('%Y-%m-%d') + 
            ", Day: " + df['day_name'] + 
            ", Month: " + df['month_name'] + 
            ", Season: " + df['season'] + 
            ", Volume_Level: " + df['volume_level'] +
            ", Tickets_Received: " + df['tickets_received'].astype(str) +
            ", Tickets_Resolved: " + df['tickets_resolved'].astype(str)
        )
        
        # Add to feature names
        text_features = [
            'day_name', 'month_name', 'season', 'temporal_description', 
            'volume_level', 'llm_text_feature'
        ]
        self.feature_names.extend(text_features)
        
        print(f"✓ Created {len(text_features)} text features for LLM adaptation")
        
        self.data = df
        return df
        
    def handle_missing_values(self):
        """Handle missing values created by lag and rolling features."""
        print(f"\n" + "=" * 60)
        print("HANDLING MISSING VALUES")
        print("=" * 60)
        
        df = self.data.copy()
        
        # Count missing values before handling
        missing_before = df.isnull().sum().sum()
        print(f"Missing values before handling: {missing_before}")
        
        # Fill missing values
        # For lag features, forward fill then backward fill
        lag_columns = [col for col in df.columns if 'lag_' in col]
        for col in lag_columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # For rolling features, forward fill
        rolling_columns = [col for col in df.columns if 'rolling_' in col]
        for col in rolling_columns:
            df[col] = df[col].fillna(method='ffill')
        
        # For difference features, fill with 0
        diff_columns = [col for col in df.columns if 'diff_' in col]
        for col in diff_columns:
            df[col] = df[col].fillna(0)
        
        # For ratio features, handle division by zero
        ratio_columns = [col for col in df.columns if 'ratio' in col]
        for col in ratio_columns:
            df[col] = df[col].fillna(0)
        
        # Count missing values after handling
        missing_after = df.isnull().sum().sum()
        print(f"Missing values after handling: {missing_after}")
        print(f"✓ Reduced missing values by {missing_before - missing_after}")
        
        self.data = df
        return df
        
    def create_visualizations(self):
        """Create visualizations of the engineered features."""
        print(f"\n" + "=" * 60)
        print("CREATING FEATURE VISUALIZATIONS")
        print("=" * 60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Engineering Analysis', fontsize=16, fontweight='bold')
        
        # 1. Temporal patterns
        daily_avg = self.data.groupby('day_of_week')['tickets_received'].mean()
        axes[0, 0].bar(range(7), daily_avg.values)
        axes[0, 0].set_title('Average Tickets by Day of Week')
        axes[0, 0].set_xlabel('Day of Week')
        axes[0, 0].set_ylabel('Average Tickets')
        axes[0, 0].set_xticks(range(7))
        axes[0, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # 2. Monthly patterns
        monthly_avg = self.data.groupby('month')['tickets_received'].mean()
        axes[0, 1].plot(monthly_avg.index, monthly_avg.values, marker='o')
        axes[0, 1].set_title('Average Tickets by Month')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Tickets')
        axes[0, 1].set_xticks(range(1, 13))
        
        # 3. Lag correlation
        lag_cols = [col for col in self.data.columns if 'tickets_received_lag_' in col][:7]
        lag_corr = self.data[['tickets_received'] + lag_cols].corr()['tickets_received'][1:]
        axes[0, 2].bar(range(len(lag_corr)), lag_corr.values)
        axes[0, 2].set_title('Lag Feature Correlations')
        axes[0, 2].set_xlabel('Lag (days)')
        axes[0, 2].set_ylabel('Correlation with Target')
        axes[0, 2].set_xticks(range(len(lag_corr)))
        axes[0, 2].set_xticklabels([f'Lag {i+1}' for i in range(len(lag_corr))])
        
        # 4. Rolling mean comparison
        sample_data = self.data.head(1000)
        axes[1, 0].plot(sample_data['date'], sample_data['tickets_received'], label='Original', alpha=0.7)
        axes[1, 0].plot(sample_data['date'], sample_data['tickets_received_rolling_mean_7'], label='7-day MA', linewidth=2)
        axes[1, 0].set_title('Original vs 7-day Moving Average')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Tickets')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Volume level distribution
        volume_counts = self.data['volume_level'].value_counts()
        axes[1, 1].pie(volume_counts.values, labels=volume_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Volume Level Distribution')
        
        # 6. Feature correlation heatmap (sample)
        feature_cols = ['tickets_received', 'tickets_received_lag_1', 'tickets_received_lag_7',
                       'tickets_received_rolling_mean_7', 'day_of_week', 'month']
        corr_matrix = self.data[feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('../results/feature_engineering_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Feature visualizations saved as 'feature_engineering_visualizations.png'")
        
    def generate_feature_summary(self):
        """Generate a summary of all engineered features."""
        print(f"\n" + "=" * 60)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 60)
        
        print(f"📊 TOTAL FEATURES CREATED: {len(self.feature_names)}")
        print(f"📊 TOTAL COLUMNS IN DATASET: {len(self.data.columns)}")
        print(f"📊 DATASET SHAPE: {self.data.shape}")
        
        # Categorize features
        temporal_features = [f for f in self.feature_names if any(x in f for x in ['year', 'month', 'day', 'week', 'quarter', 'sin', 'cos', 'is_'])]
        lag_features = [f for f in self.feature_names if 'lag_' in f]
        rolling_features = [f for f in self.feature_names if 'rolling_' in f]
        diff_features = [f for f in self.feature_names if 'diff_' in f]
        ratio_features = [f for f in self.feature_names if 'ratio' in f or 'backlog' in f]
        text_features = [f for f in self.feature_names if any(x in f for x in ['name', 'season', 'description', 'level', 'llm'])]
        
        print(f"\n📈 FEATURE CATEGORIES:")
        print(f"  • Temporal features: {len(temporal_features)}")
        print(f"  • Lag features: {len(lag_features)}")
        print(f"  • Rolling features: {len(rolling_features)}")
        print(f"  • Difference features: {len(diff_features)}")
        print(f"  • Ratio features: {len(ratio_features)}")
        print(f"  • Text features: {len(text_features)}")
        
        # Save feature list
        import json
        feature_info = {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_categories': {
                'temporal': temporal_features,
                'lag': lag_features,
                'rolling': rolling_features,
                'difference': diff_features,
                'ratio': ratio_features,
                'text': text_features
            }
        }
        
        with open('../results/feature_engineering_summary.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"\n✓ Feature summary saved to 'feature_engineering_summary.json'")
        
    def run_complete_feature_engineering(self):
        """Run the complete feature engineering pipeline."""
        print("=" * 80)
        print("STARTING FEATURE ENGINEERING")
        print("=" * 80)
        
        # Load data
        self.load_cleaned_data()
        
        # Create features
        self.create_temporal_features()
        self.create_lag_features(max_lags=14)
        self.create_rolling_features(windows=[3, 7, 14, 30])
        self.create_difference_features()
        self.create_ratio_features()
        self.create_llm_text_features()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate summary
        self.generate_feature_summary()
        
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 80)
        print("✓ All features created and processed")
        print("✓ Missing values handled")
        print("✓ Visualizations created")
        print("✓ Summary generated")
        print("✓ Ready for LLM adaptation")
        
        # Store final dataset
        self.features_data = self.data
        
        return self.data

if __name__ == "__main__":
    # Run complete feature engineering pipeline
    engineer = FeatureEngineer()
    features_data = engineer.run_complete_feature_engineering()
    
    # Save engineered data
    features_data.to_csv('../data/engineered_features_data.csv', index=False)
    print("✓ Engineered data saved to 'engineered_features_data.csv'")
