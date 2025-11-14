# COVID-19 Data Analysis Project

A comprehensive Python-based data analysis project for COVID-19 global statistics with automated visualizations and detailed reports.

## 📊 Features

- **Automated Data Loading**: Fetches latest COVID-19 data from multiple sources
- **Global Statistics**: Comprehensive overview of worldwide cases, deaths, and vaccinations
- **Top Countries Analysis**: Identifies and analyzes most affected countries
- **Time Series Trends**: Tracks progression of cases, deaths, and vaccinations over time
- **Correlation Analysis**: Examines relationships between different COVID-19 metrics
- **Professional Visualizations**: Generates high-quality charts and graphs
- **Summary Reports**: Creates detailed text-based analysis reports

## 🎯 Generated Visualizations

The project automatically generates three visualization files:

1. **covid_top_countries_analysis.png**
   - Total cases by country
   - Deaths per million population
   - Vaccination progress
   - Case fatality rates

2. **covid_time_series_analysis.png**
   - Daily new cases trends
   - Daily deaths trends
   - Cumulative cases over time
   - Vaccination progress timeline

3. **covid_correlation_analysis.png**
   - Heatmap showing correlations between metrics
   - GDP, age demographics, and healthcare infrastructure relationships

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.7 or higher
```

### Installation

1. Clone or download this project

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn
```

### Running the Analysis

**Option 1: Automatic Download (Recommended)**
```bash
python covid_analysis.py
```

**Option 2: Using Kaggle Data**

1. Download COVID-19 dataset from Kaggle:
   - [Novel Corona Virus 2019 Dataset](https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset)
   - [Corona Virus Report](https://www.kaggle.com/datasets/imdevskp/corona-virus-report)

2. Place the CSV file in the project directory

3. Edit the script and update:
```python
KAGGLE_FILE_PATH = "your_downloaded_file.csv"
```

4. Run the script:
```bash
python covid_analysis.py
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | Latest | Data manipulation and analysis |
| numpy | Latest | Numerical computations |
| matplotlib | Latest | Data visualization |
| seaborn | Latest | Statistical graphics |

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn
```

## 📁 Project Structure

```
covid-analysis/
│
├── covid_analysis.py                    # Main script
├── README.md                            # This file
│
└── outputs/
    ├── covid_top_countries_analysis.png
    ├── covid_time_series_analysis.png
    └── covid_correlation_analysis.png
```

## 🔍 Data Sources

The project supports multiple data sources:

1. **Our World in Data (OWID)**: Comprehensive global COVID-19 dataset
   - Automatic download from GitHub repository
   - Most up-to-date information

2. **Kaggle Datasets**: Community-maintained datasets
   - Reliable historical data
   - Multiple format support

3. **Local CSV Files**: Custom datasets
   - Full control over data source
   - Offline analysis capability

## 📈 Analysis Components

### 1. Global Statistics
- Total confirmed cases worldwide
- Total deaths and mortality rates
- Vaccination coverage statistics
- Case fatality rate calculations

### 2. Country-Level Analysis
- Top 15 countries by total cases
- Deaths per million population comparison
- Vaccination rate rankings
- Case fatality rate comparisons

### 3. Time Series Analysis
- Daily new cases trends (7-day moving average)
- Daily deaths trends (7-day moving average)
- Cumulative case progression
- Vaccination rollout timeline

### 4. Correlation Studies
- Relationship between cases and deaths
- Impact of GDP on health outcomes
- Age demographics correlation
- Healthcare infrastructure analysis

## 🎨 Customization

### Modify Countries for Time Series

Edit the `time_series_analysis` method call in `main()`:

```python
analysis.time_series_analysis(
    countries=['United States', 'India', 'Brazil', 'Germany', 'Japan']
)
```

### Change Number of Top Countries

Modify the parameter in `visualize_top_countries`:

```python
analysis.visualize_top_countries(n=20)
```

### Adjust Visualization Style

At the top of the script, change:

```python
plt.style.use('seaborn-v0_8-darkgrid')  # Options: 'ggplot', 'fivethirtyeight', etc.
sns.set_palette("husl")  # Options: 'Set2', 'Pastel1', 'Dark2', etc.
```

## 📊 Sample Output

```
============================================================
COVID-19 COMPREHENSIVE DATA ANALYSIS
============================================================
Loading COVID-19 data...
✓ Data loaded successfully: 450,000 records
✓ Date range: 2020-01-01 to 2024-11-14
✓ Locations: 200

Preprocessing data...
✓ Data preprocessing complete

============================================================
GLOBAL COVID-19 STATISTICS
============================================================
Total Cases: 775,000,000
Total Deaths: 7,050,000
Total Vaccinations: 13,500,000,000
People Fully Vaccinated: 5,600,000,000
Case Fatality Rate: 0.91%
============================================================
```

## ⚠️ Troubleshooting

### Issue: Data download fails

**Solution**: Use manual Kaggle download option
- Download CSV from Kaggle links provided
- Update `KAGGLE_FILE_PATH` variable
- Run script again

### Issue: Missing columns error

**Solution**: Check dataset format
- Ensure CSV has required columns (location, date, cases, deaths)
- Try different Kaggle dataset
- Check column names match expected format

### Issue: Visualization not showing

**Solution**: Check matplotlib backend
```python
import matplotlib
matplotlib.use('TkAgg')  # Add at top of script
```

### Issue: Memory error with large datasets

**Solution**: Reduce data size
```python
# Load only recent data
self.df = self.df[self.df['date'] > '2023-01-01']
```

## 📝 License

This project is open source and available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional visualization types
- More statistical analyses
- Support for additional data sources
- Performance optimizations
- Interactive dashboards

## 📧 Contact

For questions or suggestions, please open an issue in the project repository.

## 🔄 Updates

The project automatically fetches the latest data when run. For the most current statistics, simply re-run the script.

---

**Last Updated**: November 2024  
**Python Version**: 3.7+  
**Status**: Active Development
