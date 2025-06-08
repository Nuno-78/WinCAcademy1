import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from Our World in Data
def load_data():
    """Load CO2 data from Our World in Data website"""
    print("Loading data from Our World in Data...")
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    data = pd.read_csv(url)
    print(f"Data loaded: {len(data)} rows")
    return data.copy()

# QUESTION 1: What predicts high CO2 per capita?
def question1_co2_predictors(data):
    """Find what predicts CO2 emissions per capita"""
    print("\n=== QUESTION 1: CO2 Predictors ===")
    
    # Get recent data (2019-2022) and remove missing values
    recent_data = data[data['year'] >= 2019].copy()
    
    # Select columns we want to analyze
    columns = ['country', 'co2_per_capita', 'gdp_per_capita', 'population', 
              'primary_energy_consumption_per_capita', 'coal_consumption_per_capita',
              'oil_consumption_per_capita', 'gas_consumption_per_capita']
    
    # Keep only these columns and remove rows with missing data
    analysis_data = recent_data[columns].copy()
    analysis_data = analysis_data.dropna()
    
    # Calculate how each factor correlates with CO2 per capita
    predictors = ['gdp_per_capita', 'primary_energy_consumption_per_capita', 
                 'coal_consumption_per_capita', 'oil_consumption_per_capita', 
                 'gas_consumption_per_capita']
    
    correlations = {}
    for predictor in predictors:
        corr = analysis_data['co2_per_capita'].corr(analysis_data[predictor])
        correlations[predictor] = corr
        print(f"{predictor}: {corr:.3f}")
    
    # Find the strongest predictor
    strongest = max(correlations.items(), key=lambda x: abs(x[1]))
    print(f"\nStrongest predictor: {strongest[0]} (correlation: {strongest[1]:.3f})")
    
    # Create Graph 1
    plt.figure(figsize=(12, 5))
    
    # Left plot: Bar chart of correlations
    plt.subplot(1, 2, 1)
    names = [name.replace('_', ' ').title() for name in correlations.keys()]
    values = list(correlations.values())
    colors = ['red' if x < 0 else 'blue' for x in values]
    
    plt.bar(range(len(names)), values, color=colors, alpha=0.7)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Correlation with CO2 per capita')
    plt.title('What Predicts CO2 Emissions?')
    plt.grid(axis='y', alpha=0.3)
    
    # Right plot: Scatter plot of strongest predictor
    plt.subplot(1, 2, 2)
    x = analysis_data[strongest[0]]
    y = analysis_data['co2_per_capita']
    plt.scatter(x, y, alpha=0.6, color='blue')
    plt.xlabel(strongest[0].replace('_', ' ').title())
    plt.ylabel('CO2 per capita (tonnes)')
    plt.title(f'CO2 vs {strongest[0].replace("_", " ").title()}')
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    return strongest[0]

# QUESTION 2: Which countries are reducing CO2 the most?
def question2_co2_reduction(data):
    """Find countries reducing CO2 emissions the most"""
    print("\n=== QUESTION 2: CO2 Reduction Leaders ===")
    
    # Look at countries with good data coverage
    countries_with_data = []
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country].copy()
        
        # Check if country has recent data (2010-2020)
        recent_data = country_data[(country_data['year'] >= 2010) & 
                                 (country_data['year'] <= 2020)]
        
        if len(recent_data) >= 5 and not recent_data['co2_per_capita'].isna().all():
            # Get average CO2 for early period (2010-2013) and late period (2017-2020)
            early = recent_data[recent_data['year'] <= 2013]['co2_per_capita'].dropna()
            late = recent_data[recent_data['year'] >= 2017]['co2_per_capita'].dropna()
            
            if len(early) > 0 and len(late) > 0:
                early_avg = early.mean()
                late_avg = late.mean()
                
                # Calculate percentage change
                if early_avg > 0:
                    percent_change = ((late_avg - early_avg) / early_avg) * 100
                    
                    # Get population (exclude very small countries)
                    pop = recent_data['population'].dropna().mean()
                    if pop > 1000000:  # Only countries with >1M people
                        countries_with_data.append({
                            'country': country,
                            'early_co2': early_avg,
                            'late_co2': late_avg,
                            'percent_change': percent_change,
                            'population': pop
                        })
    
    # Convert to DataFrame and find top reducers
    reduction_data = pd.DataFrame(countries_with_data)
    top_reducers = reduction_data.nsmallest(15, 'percent_change')
    
    print("Top countries reducing CO2 per capita:")
    for _, row in top_reducers.head(10).iterrows():
        print(f"{row['country']}: {row['percent_change']:.1f}%")
    
    # Create Graph 2
    plt.figure(figsize=(12, 8))
    
    # Get top 15 for visualization
    countries = top_reducers['country'].head(15)
    changes = top_reducers['percent_change'].head(15)
    
    # Create horizontal bar chart
    colors = plt.cm.RdYlGn([0.8 - (i/15)*0.6 for i in range(len(countries))])
    
    plt.barh(range(len(countries)), changes, color=colors)
    plt.yticks(range(len(countries)), countries)
    plt.xlabel('Percent Change in CO2 per capita (%)')
    plt.title('Countries Reducing CO2 Emissions Most (2010-2020)')
    plt.grid(axis='x', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return top_reducers

# QUESTION 3: Which clean energy will be cheapest?
def question3_energy_future(data):
    """Predict which clean energy technology will have best prices"""
    print("\n=== QUESTION 3: Future Energy Prices ===")
    
    # Focus on clean energy consumption trends
    energy_types = {
        'Solar': 'solar_consumption_per_capita',
        'Wind': 'wind_consumption_per_capita', 
        'Nuclear': 'nuclear_consumption_per_capita',
        'Hydro': 'hydro_consumption_per_capita'
    }
    
    # Get global totals by year (sum across all countries)
    yearly_totals = data.groupby('year')[list(energy_types.values())].sum().reset_index()
    
    # Look at recent years with good data
    recent_years = yearly_totals[yearly_totals['year'] >= 2005].copy()
    
    # Calculate growth rates for each energy type
    growth_rates = {}
    
    for name, column in energy_types.items():
        if column in recent_years.columns:
            # Get data points
            years = recent_years['year'].values
            consumption = recent_years[column].values
            
            # Remove zeros and calculate average annual growth
            if len(years) > 5 and consumption.max() > 0:
                # Simple growth calculation: compare first 3 years to last 3 years
                early_avg = consumption[:3].mean()
                late_avg = consumption[-3:].mean()
                
                if early_avg > 0:
                    total_growth = (late_avg - early_avg) / early_avg
                    years_span = years[-1] - years[0]
                    annual_growth = (total_growth / years_span) * 100
                    growth_rates[name] = annual_growth
                    
                    print(f"{name}: {annual_growth:.1f}% per year")
    
    # Sort by growth rate
    sorted_growth = sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nFastest growing (likely cheapest future): {sorted_growth[0][0]}")
    
    # Create Graph 3
    plt.figure(figsize=(12, 6))
    
    # Left plot: Growth rates comparison
    plt.subplot(1, 2, 1)
    names = [item[0] for item in sorted_growth]
    rates = [item[1] for item in sorted_growth]
    colors = ['green' if x > 0 else 'red' for x in rates]
    
    plt.bar(names, rates, color=colors, alpha=0.7)
    plt.ylabel('Annual Growth Rate (%)')
    plt.title('Clean Energy Growth Rates (2005-2022)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Right plot: Recent consumption levels
    plt.subplot(1, 2, 2)
    latest_year = recent_years[recent_years['year'] == recent_years['year'].max()]
    
    consumption_levels = []
    for name, column in energy_types.items():
        if column in latest_year.columns:
            consumption_levels.append(latest_year[column].iloc[0])
        else:
            consumption_levels.append(0)
    
    plt.bar(names, consumption_levels, alpha=0.7, color='blue')
    plt.ylabel('Global Consumption (Recent Year)')
    plt.title('Current Clean Energy Scale')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sorted_growth

# MAIN ANALYSIS
def main():
    """Run the complete analysis"""
    print("=== CO2 AND ENERGY ANALYSIS ===")
    print("Data source: Our World in Data")
    
    # Load data
    data = load_data()
    
    # Run all three questions
    strongest_predictor = question1_co2_predictors(data)
    top_reducers = question2_co2_reduction(data)
    energy_ranking = question3_energy_future(data)
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"1. Biggest CO2 predictor: {strongest_predictor.replace('_', ' ').title()}")
    print(f"2. Top CO2 reducer: {top_reducers.iloc[0]['country']}")
    print(f"3. Fastest growing clean energy: {energy_ranking[0][0]}")

# Run the analysis
if __name__ == "__main__":
    main()
