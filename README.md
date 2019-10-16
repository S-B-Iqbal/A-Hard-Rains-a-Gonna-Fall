# A-Hard-Rain's-a-Gonna-Fall
Python Based implementation of OpenWeatherMap API for making a 5 day - 3 hour forecast.

## Workflow


- Wikipedia Scraping
    - Scraped list of top-50 cities in India by Population.
- OpenWeatherData Scraping
    - Created an API KEY for fetching 5 days 3 hours forecast of top 50 cities in India.
    - 'city_dataset': Contains list of Cities along with Latitude,Longitude and Country Info
    - 'merged_dataset': Contains 5 days Weather Forecast with the required matrices.
- Saving Data
    - Saved the Dataset in a serialized format.
- Data Visualization
    - Plot of Average Temperature across all cities for the 3 Hour window.
- Exploratory Data Analysis
    - Univariate Analysis
    - Multivariate Analysis
    - Correlation Plot
    - Average Temperature Plot
- Statistical Analysis
    - Objective : Impact of factors contributing to lower min_temp
    - Data preprocessing: Variable standardization.
    - Linear Regression
