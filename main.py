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
        
    def load_data(self):
        print("Loading COVID-19 data...")
        
        if KAGGLE_FILE_PATH and os.path.exists(KAGGLE_FILE_PATH):
            try:
                print(f"Loading from local file: {KAGGLE_FILE_PATH}")
                self.df = pd.read_csv(KAGGLE_FILE_PATH)
                self._standardize_columns()
                print(f"✓ Data loaded successfully: {len(self.df)} records")
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
                print(f"✓ Data loaded successfully: {len(self.df)} records")
                if 'date' in self.df.columns:
                    print(f"✓ Date range: {self.df['date'].min()} to {self.df['date'].max()}")
                if 'location' in self.df.columns:
                    print(f"✓ Locations: {self.df['location'].nunique()}")
                return True
            except Exception as e:
               
                continue
        
        return False
    
    def _standardize_columns(self):
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
        print("\nPreprocessing data...")
        print(f"Available columns: {list(self.df.columns)[:10]}...")
        
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        
        # Extended list of aggregates to exclude
        continents = ['World', 'Europe', 'Asia', 'Africa', 'North America', 
                     'South America', 'Oceania', 'European Union', 'Antarctica',
                     'High income', 'Upper middle income', 'Lower middle income', 'Low income',
                     'High-income countries', 'Upper-middle-income countries', 
                     'Lower-middle-income countries', 'Low-income countries',
                     'International']
        
        # Filter out aggregates and income groups
        mask1 = ~self.df['location'].isin(continents)
        mask2 = ~self.df['location'].str.contains('income', case=False, na=False)
        mask3 = ~self.df['location'].str.match(r'.*\(\d+\)$', na=False)
        
        self.df_countries = self.df[mask1 & mask2 & mask3].copy()
        
        print(f"Total records after filtering: {len(self.df_countries)}")
        
        latest_date = self.df['date'].max()
        print(f"Latest date in dataset: {latest_date}")
        
        self.latest_data = self.df_countries[self.df_countries['date'] == latest_date].copy()
        
        # Check if total_cases column exists and has data
        if 'total_cases' in self.latest_data.columns:
            valid_before = len(self.latest_data)
            self.latest_data = self.latest_data[
                (self.latest_data['total_cases'].notna()) & 
                (self.latest_data['total_cases'] > 0)
            ].copy()
            print(f"Records with valid case data: {len(self.latest_data)} (filtered from {valid_before})")
        else:
            print("⚠ WARNING: 'total_cases' column not found!")
            print(f"Available columns: {list(self.latest_data.columns)}")
        
        if len(self.latest_data) > 0:
            print(f"✓ Data preprocessing complete")
            print(f"✓ Countries in latest data: {len(self.latest_data)}")
            print(f"Top 5 countries by cases:")
            if 'total_cases' in self.latest_data.columns:
                top_5 = self.latest_data.nlargest(5, 'total_cases')[['location', 'total_cases']]
                for idx, row in top_5.iterrows():
                    print(f"  - {row['location']}: {row['total_cases']:,.0f}")
        else:
            print("⚠ WARNING: No valid country data found after preprocessing!")
    
    def global_statistics(self):
        print("\n" + "="*60)
        print("GLOBAL COVID-19 STATISTICS")
        print("="*60)
        
        world_data = self.df[self.df['location'] == 'World'].iloc[-1]
        
        stats = {
            'Total Cases': world_data.get('total_cases'),
            'Total Deaths': world_data.get('total_deaths'),
            'Total Vaccinations': world_data.get('total_vaccinations'),
            'People Fully Vaccinated': world_data.get('people_fully_vaccinated'),
            'Case Fatality Rate': (world_data.get('total_deaths', 0) / world_data.get('total_cases', 1) * 100) if world_data.get('total_cases') else None
        }
        
        for key, value in stats.items():
            if pd.notna(value):
                if 'Rate' in key:
                    print(f"{key}: {value:.2f}%")
                else:
                    print(f"{key}: {value:,.0f}")
        
        print("="*60)
    
    def top_countries_analysis(self, n=15):
        print(f"\nAnalyzing top {n} countries...")
        
        top_cases = self.latest_data.nlargest(n, 'total_cases')[['location', 'total_cases', 'total_deaths', 'population']]
        top_cases['deaths_per_million'] = (top_cases['total_deaths'] / top_cases['population']) * 1_000_000
        
        return top_cases
    
    def visualize_top_countries(self, n=15):
        print("\nGenerating visualizations...")
        
        if len(self.latest_data) == 0:
            print("⚠ No data available for visualization")
            return
        
        # Get top N countries with valid data
        top_data = self.latest_data.nlargest(min(n, len(self.latest_data)), 'total_cases').copy()
        
        print(f"Visualizing data for {len(top_data)} countries")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('COVID-19: Top Countries Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Total Cases
        ax1 = axes[0, 0]
        if len(top_data) > 0 and 'total_cases' in top_data.columns:
            countries = top_data['location'].values
            cases = top_data['total_cases'].values / 1_000_000
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(countries)))
            bars1 = ax1.barh(countries, cases, color=colors)
            ax1.set_xlabel('Total Cases (Millions)', fontsize=11)
            ax1.set_title('Total COVID-19 Cases by Country', fontsize=12, fontweight='bold')
            ax1.invert_yaxis()
            for i, v in enumerate(cases):
                ax1.text(v, i, f' {v:.1f}M', va='center', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No valid data available', ha='center', va='center', transform=ax1.transAxes)
        
        # Plot 2: Deaths per Million (with valid data check)
        ax2 = axes[0, 1]
        if 'total_deaths_per_million' in self.latest_data.columns:
            valid_death_data = self.latest_data[
                (self.latest_data['total_deaths_per_million'].notna()) & 
                (self.latest_data['total_deaths_per_million'] > 0)
            ]
            if len(valid_death_data) > 0:
                top_death_rate = valid_death_data.nlargest(min(n, len(valid_death_data)), 'total_deaths_per_million')
                countries_dr = top_death_rate['location'].values
                death_rate = top_death_rate['total_deaths_per_million'].values
                colors_dr = plt.cm.Oranges(np.linspace(0.4, 0.9, len(countries_dr)))
                ax2.barh(countries_dr, death_rate, color=colors_dr)
                ax2.set_xlabel('Deaths per Million', fontsize=11)
                ax2.set_title('COVID-19 Deaths per Million Population', fontsize=12, fontweight='bold')
                ax2.invert_yaxis()
                for i, v in enumerate(death_rate):
                    ax2.text(v, i, f' {v:.0f}', va='center', fontsize=9)
            else:
                ax2.text(0.5, 0.5, 'No valid death rate data', ha='center', va='center', transform=ax2.transAxes)
        elif 'total_deaths' in top_data.columns and 'population' in top_data.columns:
            # Calculate deaths per million manually
            valid_death_data = top_data[
                (top_data['total_deaths'].notna()) & 
                (top_data['population'].notna()) &
                (top_data['total_deaths'] > 0) &
                (top_data['population'] > 0)
            ].copy()
            if len(valid_death_data) > 0:
                valid_death_data['deaths_per_million'] = (valid_death_data['total_deaths'] / valid_death_data['population']) * 1_000_000
                top_death_rate = valid_death_data.nlargest(min(n, len(valid_death_data)), 'deaths_per_million')
                countries_dr = top_death_rate['location'].values
                death_rate = top_death_rate['deaths_per_million'].values
                colors_dr = plt.cm.Oranges(np.linspace(0.4, 0.9, len(countries_dr)))
                ax2.barh(countries_dr, death_rate, color=colors_dr)
                ax2.set_xlabel('Deaths per Million', fontsize=11)
                ax2.set_title('COVID-19 Deaths per Million Population', fontsize=12, fontweight='bold')
                ax2.invert_yaxis()
                for i, v in enumerate(death_rate):
                    ax2.text(v, i, f' {v:.0f}', va='center', fontsize=9)
            else:
                ax2.text(0.5, 0.5, 'No valid death data', ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'Death data not available', ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Vaccination Progress (with valid data check)
        ax3 = axes[1, 0]
        if 'people_fully_vaccinated_per_hundred' in self.latest_data.columns:
            valid_vacc_data = self.latest_data[
                (self.latest_data['people_fully_vaccinated_per_hundred'].notna()) & 
                (self.latest_data['people_fully_vaccinated_per_hundred'] > 0)
            ]
            if len(valid_vacc_data) > 0:
                top_vacc = valid_vacc_data.nlargest(min(n, len(valid_vacc_data)), 'people_fully_vaccinated_per_hundred')
                countries_v = top_vacc['location'].values
                vacc_rate = top_vacc['people_fully_vaccinated_per_hundred'].values
                colors_v = plt.cm.Greens(np.linspace(0.4, 0.9, len(countries_v)))
                ax3.barh(countries_v, vacc_rate, color=colors_v)
                ax3.set_xlabel('% Fully Vaccinated', fontsize=11)
                ax3.set_title('Vaccination Progress (% Fully Vaccinated)', fontsize=12, fontweight='bold')
                ax3.invert_yaxis()
                for i, v in enumerate(vacc_rate):
                    ax3.text(v, i, f' {v:.1f}%', va='center', fontsize=9)
            else:
                ax3.text(0.5, 0.5, 'No vaccination data', ha='center', va='center', transform=ax3.transAxes)
        else:
            # Try alternative column names
            alt_vacc_cols = ['people_vaccinated_per_hundred', 'total_vaccinations_per_hundred']
            found = False
            for col in alt_vacc_cols:
                if col in self.latest_data.columns:
                    valid_vacc_data = self.latest_data[
                        (self.latest_data[col].notna()) & 
                        (self.latest_data[col] > 0)
                    ]
                    if len(valid_vacc_data) > 0:
                        top_vacc = valid_vacc_data.nlargest(min(n, len(valid_vacc_data)), col)
                        countries_v = top_vacc['location'].values
                        vacc_rate = top_vacc[col].values
                        colors_v = plt.cm.Greens(np.linspace(0.4, 0.9, len(countries_v)))
                        ax3.barh(countries_v, vacc_rate, color=colors_v)
                        ax3.set_xlabel('Vaccination Rate (%)', fontsize=11)
                        ax3.set_title(f'Vaccination Progress ({col.replace("_", " ").title()})', fontsize=12, fontweight='bold')
                        ax3.invert_yaxis()
                        for i, v in enumerate(vacc_rate):
                            ax3.text(v, i, f' {v:.1f}%', va='center', fontsize=9)
                        found = True
                        break
            if not found:
                ax3.text(0.5, 0.5, 'Vaccination data not available', ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Case Fatality Rate (using top countries by cases)
        ax4 = axes[1, 1]
        if 'total_deaths' in top_data.columns:
            top_cfr = top_data.copy()
            top_cfr['cfr'] = (top_cfr['total_deaths'] / top_cfr['total_cases']) * 100
            top_cfr = top_cfr[top_cfr['cfr'].notna()].sort_values('cfr', ascending=False)[:n]
            
            if len(top_cfr) > 0:
                countries_cfr = top_cfr['location'].values
                cfr = top_cfr['cfr'].values
                colors_cfr = plt.cm.Purples(np.linspace(0.4, 0.9, len(countries_cfr)))
                ax4.barh(countries_cfr, cfr, color=colors_cfr)
                ax4.set_xlabel('Case Fatality Rate (%)', fontsize=11)
                ax4.set_title('Case Fatality Rate (Top Countries)', fontsize=12, fontweight='bold')
                ax4.invert_yaxis()
                for i, v in enumerate(cfr):
                    ax4.text(v, i, f' {v:.2f}%', va='center', fontsize=9)
            else:
                ax4.text(0.5, 0.5, 'No valid CFR data', ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'Death data not available', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig('covid_top_countries_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: covid_top_countries_analysis.png")
        plt.show()
    
    def time_series_analysis(self, countries=['United States', 'India', 'Brazil', 'United Kingdom', 'France']):
        print("\nGenerating time series analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('COVID-19 Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Check which countries exist in the dataset
        available_countries = [c for c in countries if c in self.df_countries['location'].unique()]
        
        if not available_countries:
            print("⚠ None of the specified countries found. Using top 5 countries by cases...")
            available_countries = self.latest_data.nlargest(5, 'total_cases')['location'].tolist()
        
        df_selected = self.df_countries[self.df_countries['location'].isin(available_countries)]
        
        # Plot 1: Daily New Cases
        ax1 = axes[0, 0]
        for country in available_countries:
            country_data = df_selected[df_selected['location'] == country]
            if 'new_cases_smoothed' in country_data.columns and country_data['new_cases_smoothed'].notna().any():
                ax1.plot(country_data['date'], country_data['new_cases_smoothed'], 
                        label=country, linewidth=2, alpha=0.8)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Daily New Cases (7-day avg)', fontsize=11)
        ax1.set_title('Daily New Cases Trend', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Daily Deaths
        ax2 = axes[0, 1]
        for country in available_countries:
            country_data = df_selected[df_selected['location'] == country]
            if 'new_deaths_smoothed' in country_data.columns and country_data['new_deaths_smoothed'].notna().any():
                ax2.plot(country_data['date'], country_data['new_deaths_smoothed'], 
                        label=country, linewidth=2, alpha=0.8)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Daily Deaths (7-day avg)', fontsize=11)
        ax2.set_title('Daily Deaths Trend', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative Cases
        ax3 = axes[1, 0]
        for country in available_countries:
            country_data = df_selected[df_selected['location'] == country]
            if 'total_cases' in country_data.columns and country_data['total_cases'].notna().any():
                ax3.plot(country_data['date'], country_data['total_cases'] / 1_000_000, 
                        label=country, linewidth=2, alpha=0.8)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Total Cases (Millions)', fontsize=11)
        ax3.set_title('Cumulative Cases Over Time', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Vaccination Progress
        ax4 = axes[1, 1]
        for country in available_countries:
            country_data = df_selected[df_selected['location'] == country]
            if 'people_fully_vaccinated_per_hundred' in country_data.columns and country_data['people_fully_vaccinated_per_hundred'].notna().any():
                ax4.plot(country_data['date'], country_data['people_fully_vaccinated_per_hundred'], 
                        label=country, linewidth=2, alpha=0.8)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_ylabel('% Fully Vaccinated', fontsize=11)
        ax4.set_title('Vaccination Progress Over Time', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('covid_time_series_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: covid_time_series_analysis.png")
        plt.show()
    
    def correlation_analysis(self):
        print("\nGenerating correlation analysis...")
        
        cols_of_interest = ['total_cases_per_million', 'total_deaths_per_million', 
                           'people_fully_vaccinated_per_hundred', 'gdp_per_capita', 
                           'aged_65_older', 'hospital_beds_per_thousand']
        
        # Only use columns that exist in the dataset
        available_cols = [col for col in cols_of_interest if col in self.latest_data.columns]
        
        if len(available_cols) >= 2:
            corr_data = self.latest_data[available_cols].dropna()
            
            if len(corr_data) > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = corr_data.corr()
                
                sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
                
                ax.set_title('Correlation Matrix: COVID-19 Metrics', fontsize=14, fontweight='bold', pad=20)
                
                plt.tight_layout()
                plt.savefig('covid_correlation_analysis.png', dpi=300, bbox_inches='tight')
                print("✓ Saved: covid_correlation_analysis.png")
                plt.show()
            else:
                print("⚠ Not enough data for correlation analysis")
        else:
            print("⚠ Not enough columns available for correlation analysis")
    
    def generate_summary_report(self):
        print("\n" + "="*60)
        print("COVID-19 DATA ANALYSIS SUMMARY REPORT")
        print("="*60)
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        print("\nTOP 10 COUNTRIES BY TOTAL CASES:")
        print("-" * 60)
        top_10 = self.latest_data.nlargest(10, 'total_cases')[['location', 'total_cases', 'total_deaths']]
        for idx, row in top_10.iterrows():
            deaths_val = row['total_deaths'] if pd.notna(row['total_deaths']) else 0
            print(f"{row['location']:20s} | Cases: {row['total_cases']:>12,.0f} | Deaths: {deaths_val:>10,.0f}")
        
        print("\nTOP 10 COUNTRIES BY VACCINATION RATE:")
        print("-" * 60)
        if 'people_fully_vaccinated_per_hundred' in self.latest_data.columns:
            top_vacc = self.latest_data[self.latest_data['people_fully_vaccinated_per_hundred'].notna()].nlargest(10, 'people_fully_vaccinated_per_hundred')[['location', 'people_fully_vaccinated_per_hundred']]
            for idx, row in top_vacc.iterrows():
                print(f"{row['location']:20s} | Fully Vaccinated: {row['people_fully_vaccinated_per_hundred']:>6.2f}%")
        else:
            print("Vaccination data not available")
        
        print("\n" + "="*60)
        print("Analysis complete! All visualizations have been saved.")
        print("="*60)


def main():
    print("="*60)
    print("COVID-19 COMPREHENSIVE DATA ANALYSIS")
    print("="*60)
    
    analysis = CovidAnalysis()
    
    if not analysis.load_data():
        print("Failed to load data. Exiting...")
        return
    
    analysis.preprocess_data()
    analysis.global_statistics()
    analysis.visualize_top_countries(n=15)
    analysis.time_series_analysis()
    analysis.correlation_analysis()
    analysis.generate_summary_report()
    
    
if __name__ == "__main__":
    main()
