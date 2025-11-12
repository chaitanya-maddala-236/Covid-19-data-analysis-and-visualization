import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

KAGGLE_FILE_PATH = None

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CovidAnalysis:
    def __init__(self):
        self.df = None
        self.latest_data = None
        self.df_countries = None
        
    def load_data(self):
        """Load COVID-19 data from various sources"""
        print("Loading COVID-19 data...")
        
        if KAGGLE_FILE_PATH and os.path.exists(KAGGLE_FILE_PATH):
            try:
                print(f"Loading from local file: {KAGGLE_FILE_PATH}")
                self.df = pd.read_csv(KAGGLE_FILE_PATH)
                self._standardize_columns()
                print(f"✓ Data loaded successfully: {len(self.df):,} records")
                return True
            except Exception as e:
                print(f"✗ Failed to load local file: {e}")
        
        urls = [
            "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
            "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        ]
        
        for url in urls:
            try:
                print(f"Downloading from: {url[:50]}...")
                self.df = pd.read_csv(url)
                self._standardize_columns()
                print(f"✓ Data loaded successfully: {len(self.df):,} records")
                if 'date' in self.df.columns:
                    print(f"✓ Date range: {self.df['date'].min()} to {self.df['date'].max()}")
                if 'location' in self.df.columns:
                    print(f"✓ Locations: {self.df['location'].nunique()}")
                return True
            except Exception as e:
                print(f"✗ Failed: {str(e)[:100]}")
                continue
        
        self._print_manual_instructions()
        return False
    
    
    def _standardize_columns(self):
        """Standardize column names across different data sources"""
        column_mapping = {
            'Country': 'location',
            'Country/Region': 'location',
            'Country_Region': 'location',
            'Date': 'date',
            'Confirmed': 'total_cases',
            'Deaths': 'total_deaths',
            'Recovered': 'total_recovered',
            'Active': 'active_cases'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in self.df.columns:
                self.df.rename(columns={old_name: new_name}, inplace=True)
    
    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        print("\nPreprocessing data...")
        print(f"Available columns: {', '.join(list(self.df.columns)[:15])}...")
        
        # Convert date column
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        
        # Check what locations we have
        unique_locations = self.df['location'].unique()
        print(f"Sample locations: {list(unique_locations[:10])}")
        
        # Extended list of aggregates to exclude
        aggregates = [
            'World', 'Europe', 'Asia', 'Africa', 'North America', 
            'South America', 'Oceania', 'European Union', 'Antarctica',
            'High income', 'Upper middle income', 'Lower middle income', 'Low income',
            'High-income countries', 'Upper-middle-income countries', 
            'Lower-middle-income countries', 'Low-income countries',
            'International', 'European', 'OWID'
        ]
        
        # More lenient filtering - only exclude known aggregates
        mask1 = ~self.df['location'].isin(aggregates)
        mask2 = ~self.df['location'].str.contains('income', case=False, na=False)
        # Remove the regex filter that was too aggressive
        
        self.df_countries = self.df[mask1 & mask2].copy()
        
        print(f"Total records after filtering: {len(self.df_countries):,}")
        print(f"Unique countries: {self.df_countries['location'].nunique()}")
        
        # Check if we have any data
        if len(self.df_countries) == 0:
            print("\n⚠ WARNING: All data was filtered out!")
            print("Using all data without filtering...")
            self.df_countries = self.df.copy()
        
        # Get latest data
        latest_date = self.df_countries['date'].max()
        print(f"Latest date in dataset: {latest_date}")
        
        self.latest_data = self.df_countries[self.df_countries['date'] == latest_date].copy()
        print(f"Records on latest date: {len(self.latest_data)}")
        
        # Ensure we have valid case data
        if 'total_cases' in self.latest_data.columns:
            valid_before = len(self.latest_data)
            self.latest_data = self.latest_data[
                (self.latest_data['total_cases'].notna()) & 
                (self.latest_data['total_cases'] > 0)
            ].copy()
            print(f"Countries with valid case data: {len(self.latest_data)} (filtered from {valid_before})")
            
            # If still no data, be even more lenient
            if len(self.latest_data) == 0:
                print("\n⚠ No data with filtering. Trying without case filter...")
                self.latest_data = self.df_countries[self.df_countries['date'] == latest_date].copy()
                # Just remove obvious non-countries
                world_only = self.latest_data[~self.latest_data['location'].isin(['World'])]
                if len(world_only) > 0:
                    self.latest_data = world_only
        else:
            print("⚠ WARNING: 'total_cases' column not found!")
            print(f"Available columns: {list(self.latest_data.columns)}")
            return
        
        if len(self.latest_data) > 0:
            print(f"✓ Data preprocessing complete")
            print(f"✓ Countries in latest data: {len(self.latest_data)}")
            print(f"\nTop 5 countries by cases:")
            top_5 = self.latest_data.nlargest(5, 'total_cases')[['location', 'total_cases']]
            for idx, row in top_5.iterrows():
                print(f"  {row['location']:20s}: {row['total_cases']:>12,.0f}")
        else:
            print("⚠ WARNING: No valid country data found after preprocessing!")
            print(f"Check your data structure. Column names: {list(self.df.columns[:20])}")
    
    def global_statistics(self):
        """Display global COVID-19 statistics"""
        print("\n" + "="*70)
        print("GLOBAL COVID-19 STATISTICS")
        print("="*70)
        
        world_data = self.df[self.df['location'] == 'World']
        
        if len(world_data) == 0:
            print("⚠ World data not available")
            return
        
        world_latest = world_data.iloc[-1]
        
        stats = [
            ('Total Cases', world_latest.get('total_cases'), False),
            ('Total Deaths', world_latest.get('total_deaths'), False),
            ('Total Vaccinations', world_latest.get('total_vaccinations'), False),
            ('People Fully Vaccinated', world_latest.get('people_fully_vaccinated'), False),
            ('Case Fatality Rate', 
             (world_latest.get('total_deaths', 0) / world_latest.get('total_cases', 1) * 100) 
             if world_latest.get('total_cases') else None, True)
        ]
        
        for label, value, is_percentage in stats:
            if pd.notna(value) and value > 0:
                if is_percentage:
                    print(f"{label:30s}: {value:>10.2f}%")
                else:
                    print(f"{label:30s}: {value:>15,.0f}")
        
        print("="*70)
    
    def visualize_top_countries(self, n=15):
        """Generate comprehensive visualizations for top countries"""
        print(f"\nGenerating visualizations for top {n} countries...")
        
        if len(self.latest_data) == 0:
            print("⚠ No data available for visualization")
            return
        
        top_data = self.latest_data.nlargest(min(n, len(self.latest_data)), 'total_cases').copy()
        print(f"Visualizing data for {len(top_data)} countries")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('COVID-19: Top Countries Analysis', fontsize=18, fontweight='bold', y=0.995)
        
        # Plot 1: Total Cases
        self._plot_total_cases(axes[0, 0], top_data, n)
        
        # Plot 2: Deaths per Million
        self._plot_deaths_per_million(axes[0, 1], n)
        
        # Plot 3: Vaccination Progress
        self._plot_vaccination_progress(axes[1, 0], n)
        
        # Plot 4: Case Fatality Rate
        self._plot_case_fatality_rate(axes[1, 1], top_data, n)
        
        plt.tight_layout()
        filename = 'covid_top_countries_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.show()
    
    def _plot_total_cases(self, ax, top_data, n):
        """Plot total cases by country"""
        if 'total_cases' in top_data.columns and len(top_data) > 0:
            countries = top_data['location'].values
            cases = top_data['total_cases'].values / 1_000_000
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(countries)))
            ax.barh(countries, cases, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Total Cases (Millions)', fontsize=12, fontweight='bold')
            ax.set_title('Total COVID-19 Cases by Country', fontsize=13, fontweight='bold', pad=10)
            ax.invert_yaxis()
            for i, v in enumerate(cases):
                ax.text(v + v*0.02, i, f'{v:.2f}M', va='center', fontsize=10, fontweight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'No valid data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _plot_deaths_per_million(self, ax, n):
        """Plot deaths per million population"""
        if 'total_deaths_per_million' in self.latest_data.columns:
            valid_data = self.latest_data[
                (self.latest_data['total_deaths_per_million'].notna()) & 
                (self.latest_data['total_deaths_per_million'] > 0)
            ]
            if len(valid_data) > 0:
                top_death_rate = valid_data.nlargest(min(n, len(valid_data)), 'total_deaths_per_million')
                self._create_bar_chart(ax, top_death_rate, 'total_deaths_per_million', 
                                      'Deaths per Million', 'Oranges', 
                                      'COVID-19 Deaths per Million Population', '{:.0f}')
                return
        
        # Fallback: Calculate manually
        valid_data = self.latest_data[
            (self.latest_data['total_deaths'].notna()) & 
            (self.latest_data['population'].notna()) &
            (self.latest_data['total_deaths'] > 0) &
            (self.latest_data['population'] > 0)
        ].copy()
        
        if len(valid_data) > 0:
            valid_data['deaths_per_million'] = (valid_data['total_deaths'] / valid_data['population']) * 1_000_000
            top_death_rate = valid_data.nlargest(min(n, len(valid_data)), 'deaths_per_million')
            self._create_bar_chart(ax, top_death_rate, 'deaths_per_million', 
                                  'Deaths per Million', 'Oranges', 
                                  'COVID-19 Deaths per Million Population', '{:.0f}')
        else:
            ax.text(0.5, 0.5, 'Death data not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _plot_vaccination_progress(self, ax, n):
        """Plot vaccination progress"""
        vacc_cols = ['people_fully_vaccinated_per_hundred', 'people_vaccinated_per_hundred', 
                     'total_vaccinations_per_hundred']
        
        for col in vacc_cols:
            if col in self.latest_data.columns:
                valid_data = self.latest_data[
                    (self.latest_data[col].notna()) & 
                    (self.latest_data[col] > 0)
                ]
                if len(valid_data) > 0:
                    top_vacc = valid_data.nlargest(min(n, len(valid_data)), col)
                    title = 'Vaccination Progress (% Fully Vaccinated)' if 'fully' in col else 'Vaccination Rate'
                    self._create_bar_chart(ax, top_vacc, col, '% Vaccinated', 'Greens', 
                                          title, '{:.1f}%')
                    return
        
        ax.text(0.5, 0.5, 'Vaccination data not available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
    
    def _plot_case_fatality_rate(self, ax, top_data, n):
        """Plot case fatality rate"""
        if 'total_deaths' in top_data.columns and 'total_cases' in top_data.columns:
            cfr_data = top_data.copy()
            cfr_data['cfr'] = (cfr_data['total_deaths'] / cfr_data['total_cases']) * 100
            cfr_data = cfr_data[cfr_data['cfr'].notna()].sort_values('cfr', ascending=False).head(n)
            
            if len(cfr_data) > 0:
                self._create_bar_chart(ax, cfr_data, 'cfr', 'Case Fatality Rate (%)', 
                                      'Purples', 'Case Fatality Rate (Top Countries)', '{:.2f}%')
            else:
                ax.text(0.5, 0.5, 'No valid CFR data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'Death data not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _create_bar_chart(self, ax, data, value_col, xlabel, colormap, title, value_format):
        """Helper function to create consistent bar charts"""
        countries = data['location'].values
        values = data[value_col].values
        colors = plt.cm.get_cmap(colormap)(np.linspace(0.4, 0.9, len(countries)))
        ax.barh(countries, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.invert_yaxis()
        for i, v in enumerate(values):
            ax.text(v + v*0.02, i, f' {value_format.format(v)}', va='center', 
                   fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    def time_series_analysis(self, countries=None):
        """Generate time series visualizations"""
        print("\nGenerating time series analysis...")
        
        if countries is None:
            countries = ['United States', 'India', 'Brazil', 'United Kingdom', 'France']
        
        # Find available countries
        available = [c for c in countries if c in self.df_countries['location'].unique()]
        
        if not available:
            print("⚠ Specified countries not found. Using top 5 countries by cases...")
            available = self.latest_data.nlargest(5, 'total_cases')['location'].tolist()
        
        print(f"Analyzing: {', '.join(available)}")
        
        df_selected = self.df_countries[self.df_countries['location'].isin(available)].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('COVID-19 Time Series Analysis', fontsize=18, fontweight='bold', y=0.995)
        
        # Plot 1: Daily New Cases
        self._plot_time_series(axes[0, 0], df_selected, available, 'new_cases_smoothed', 
                              'Daily New Cases (7-day avg)', 'Daily New Cases Trend')
        
        # Plot 2: Daily Deaths
        self._plot_time_series(axes[0, 1], df_selected, available, 'new_deaths_smoothed', 
                              'Daily Deaths (7-day avg)', 'Daily Deaths Trend')
        
        # Plot 3: Cumulative Cases
        self._plot_time_series_cumulative(axes[1, 0], df_selected, available, 'total_cases', 
                                         'Total Cases (Millions)', 'Cumulative Cases Over Time')
        
        # Plot 4: Vaccination Progress
        self._plot_time_series(axes[1, 1], df_selected, available, 'people_fully_vaccinated_per_hundred', 
                              '% Fully Vaccinated', 'Vaccination Progress Over Time')
        
        plt.tight_layout()
        filename = 'covid_time_series_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.show()
    
    def _plot_time_series(self, ax, df, countries, column, ylabel, title):
        """Helper for time series plots"""
        plotted = False
        for country in countries:
            data = df[df['location'] == country]
            if column in data.columns and data[column].notna().any():
                ax.plot(data['date'], data[column], label=country, linewidth=2.5, alpha=0.85)
                plotted = True
        
        if plotted:
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.legend(fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, f'No data available for {column}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _plot_time_series_cumulative(self, ax, df, countries, column, ylabel, title):
        """Helper for cumulative time series plots"""
        plotted = False
        for country in countries:
            data = df[df['location'] == country]
            if column in data.columns and data[column].notna().any():
                ax.plot(data['date'], data[column] / 1_000_000, label=country, linewidth=2.5, alpha=0.85)
                plotted = True
        
        if plotted:
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.legend(fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, f'No data available for {column}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def correlation_analysis(self):
        """Generate correlation matrix heatmap"""
        print("\nGenerating correlation analysis...")
        
        cols_of_interest = [
            'total_cases_per_million', 'total_deaths_per_million', 
            'people_fully_vaccinated_per_hundred', 'gdp_per_capita', 
            'aged_65_older', 'hospital_beds_per_thousand',
            'population_density', 'median_age', 'cardiovasc_death_rate',
            'diabetes_prevalence', 'life_expectancy'
        ]
        
        # Find available columns
        available_cols = [col for col in cols_of_interest if col in self.latest_data.columns]
        
        print(f"Found {len(available_cols)} columns for correlation analysis")
        
        if len(available_cols) < 2:
            print("⚠ Not enough columns available for correlation analysis")
            print(f"Available columns: {available_cols}")
            return
        
        # Create correlation data with at least 20 valid rows
        corr_data = self.latest_data[available_cols].dropna(thresh=len(available_cols)//2)
        
        if len(corr_data) < 20:
            print(f"⚠ Only {len(corr_data)} countries have sufficient data for correlation")
            # Try with more lenient threshold
            corr_data = self.latest_data[available_cols].dropna(thresh=2)
        
        if len(corr_data) >= 2:
            # Remove columns that are all NaN
            corr_data = corr_data.dropna(axis=1, how='all')
            
            if len(corr_data.columns) >= 2:
                fig, ax = plt.subplots(figsize=(12, 10))
                correlation_matrix = corr_data.corr()
                
                sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, linewidths=1.5, cbar_kws={"shrink": 0.8}, 
                           ax=ax, vmin=-1, vmax=1)
                
                ax.set_title('Correlation Matrix: COVID-19 and Socioeconomic Metrics', 
                           fontsize=14, fontweight='bold', pad=20)
                
                # Improve label readability
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
                
                plt.tight_layout()
                filename = 'covid_correlation_analysis.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {filename} ({len(corr_data)} countries, {len(correlation_matrix.columns)} metrics)")
                plt.show()
            else:
                print("⚠ Not enough valid columns after cleaning")
        else:
            print("⚠ Not enough data for correlation analysis")
    
    def generate_summary_report(self):
        """Generate comprehensive text summary report"""
        print("\n" + "="*70)
        print("COVID-19 DATA ANALYSIS SUMMARY REPORT")
        print("="*70)
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data as of: {self.df_countries['date'].max().strftime('%Y-%m-%d')}")
        print("="*70)
        
        # Top 10 by cases
        print("\n📊 TOP 10 COUNTRIES BY TOTAL CASES:")
        print("-" * 70)
        if 'total_cases' in self.latest_data.columns:
            top_10_cases = self.latest_data.nlargest(10, 'total_cases')
            print(f"{'Rank':<6}{'Country':<25}{'Total Cases':>15}{'Total Deaths':>15}")
            print("-" * 70)
            for i, (idx, row) in enumerate(top_10_cases.iterrows(), 1):
                cases = row['total_cases']
                deaths = row.get('total_deaths', 0) if pd.notna(row.get('total_deaths')) else 0
                print(f"{i:<6}{row['location']:<25}{cases:>15,.0f}{deaths:>15,.0f}")
        else:
            print("Data not available")
        
        # Top 10 by vaccination rate
        print("\n💉 TOP 10 COUNTRIES BY VACCINATION RATE:")
        print("-" * 70)
        vacc_col = None
        for col in ['people_fully_vaccinated_per_hundred', 'people_vaccinated_per_hundred']:
            if col in self.latest_data.columns:
                vacc_col = col
                break
        
        if vacc_col:
            valid_vacc = self.latest_data[self.latest_data[vacc_col].notna()]
            if len(valid_vacc) > 0:
                top_10_vacc = valid_vacc.nlargest(10, vacc_col)
                print(f"{'Rank':<6}{'Country':<25}{'Vaccination Rate':>20}")
                print("-" * 70)
                for i, (idx, row) in enumerate(top_10_vacc.iterrows(), 1):
                    rate = row[vacc_col]
                    print(f"{i:<6}{row['location']:<25}{rate:>19.2f}%")
            else:
                print("No vaccination data available")
        else:
            print("Vaccination data not available in dataset")
        
        # Deaths per million
        print("\n☠️  TOP 10 COUNTRIES BY DEATHS PER MILLION:")
        print("-" * 70)
        if 'total_deaths_per_million' in self.latest_data.columns:
            valid_deaths = self.latest_data[self.latest_data['total_deaths_per_million'].notna()]
            if len(valid_deaths) > 0:
                top_10_deaths = valid_deaths.nlargest(10, 'total_deaths_per_million')
                print(f"{'Rank':<6}{'Country':<25}{'Deaths per Million':>20}")
                print("-" * 70)
                for i, (idx, row) in enumerate(top_10_deaths.iterrows(), 1):
                    dpm = row['total_deaths_per_million']
                    print(f"{i:<6}{row['location']:<25}{dpm:>20,.0f}")
            else:
                print("No death rate data available")
        else:
            print("Death rate data not available in dataset")
        
        print("\n" + "="*70)
        print("✓ Analysis complete! All visualizations have been saved.")
        print("="*70)
        print("\nGenerated files:")
        print("  • covid_top_countries_analysis.png")
        print("  • covid_time_series_analysis.png")
        print("  • covid_correlation_analysis.png")
        print("="*70)


def main():
    """Main execution function"""
    print("="*70)
    print("         COVID-19 COMPREHENSIVE DATA ANALYSIS         ")
    print("="*70)
    
    analysis = CovidAnalysis()
    
    # Load data
    if not analysis.load_data():
        print("\n❌ Failed to load data. Please follow the instructions above.")
        return
    
    # Preprocess
    analysis.preprocess_data()
    
    if analysis.latest_data is None or len(analysis.latest_data) == 0:
        print("\n❌ No valid data available for analysis. Exiting...")
        return
    
    # Run analyses
    analysis.global_statistics()
    analysis.visualize_top_countries(n=15)
    analysis.time_series_analysis()
    analysis.correlation_analysis()
    analysis.generate_summary_report()
    
    print("\n✅ All analyses completed successfully!")
    print("📁 Check the current directory for saved PNG visualizations")


if __name__ == "__main__":
    main()
