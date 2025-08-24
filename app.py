import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

# Load the saved models
regressor_pipeline = joblib.load("regressor_pipeline.pkl")
classifier_pipeline = joblib.load("classifier_pipeline.pkl")

# Function to predict AQI using only Country and City
def predict_aqi_with_defaults(country, city, dataset):
    matched_data = dataset[(dataset['Country'] == country) & (dataset['City'] == city)]
    if matched_data.empty:
        raise ValueError("No matching data found for the specified Country and City.")

    row = matched_data.iloc[0]

    input_data = pd.DataFrame({
        'Country': [country],
        'City': [city],
        'CO AQI Value': [row.get('CO AQI Value', dataset['CO AQI Value'].mean())],
        'Ozone AQI Value': [row.get('Ozone AQI Value', dataset['Ozone AQI Value'].mean())],
        'NO2 AQI Value': [row.get('NO2 AQI Value', dataset['NO2 AQI Value'].mean())],
        'PM2.5 AQI Value': [row.get('PM2.5 AQI Value', dataset['PM2.5 AQI Value'].mean())],
        'lat': [row.get('lat', dataset['lat'].mean())],
        'lng': [row.get('lng', dataset['lng'].mean())]
    })

    aqi_value = regressor_pipeline.predict(input_data)[0]
    aqi_category = classifier_pipeline.predict(input_data)[0]
    return aqi_value, aqi_category

# Streamlit Web Application
def main():
    st.title("AQI Prediction System")

    st.sidebar.title("Options")
    page = st.sidebar.selectbox("Choose a page:", ["Home", "Analysis", "Visualization"])

    if page == "Home":
        st.header("Home: Upload and Predict AQI")
        upload_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

        if upload_file is not None:
            dataset = pd.read_csv(upload_file)
            required_columns = ['Country', 'City', 'CO AQI Value', 'Ozone AQI Value', 
                                'NO2 AQI Value', 'PM2.5 AQI Value', 'lat', 'lng']

            if not all(col in dataset.columns for col in required_columns):
                st.error(f"Dataset is missing required columns: {', '.join(required_columns)}")
                return

            st.success("Dataset loaded successfully!")

            st.header("Predict AQI")
            country = st.text_input("Enter the Country:")
            city = st.text_input("Enter the City:")

            if st.button("Predict AQI"):
                try:
                    aqi_value, aqi_category = predict_aqi_with_defaults(country, city, dataset)
                    st.success(f"Predicted AQI Value: {aqi_value:.2f}")
                    st.info(f"Predicted AQI Category: {aqi_category}")
                except ValueError as e:
                    st.error(str(e))

    elif page == "Analysis":
        st.header("Visualize AQI Analysis")
        upload_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"], key="analysis")

        if upload_file is not None:
            
            dataset = pd.read_csv(upload_file)

            selected_countries = st.multiselect("Select Countries:", dataset['Country'].unique(), max_selections=5)

            if selected_countries:
                analysis_data = dataset[dataset['Country'].isin(selected_countries)]

                st.subheader("Distribution of Pollutants by Country")
                for country in selected_countries:
                    country_data = analysis_data[analysis_data['Country'] == country]

                    pollutant_totals = {
                        'CO AQI Value': country_data['CO AQI Value'].sum(),
                        'Ozone AQI Value': country_data['Ozone AQI Value'].sum(),
                        'NO2 AQI Value': country_data['NO2 AQI Value'].sum(),
                        'PM2.5 AQI Value': country_data['PM2.5 AQI Value'].sum()
                    }

                    fig, ax = plt.subplots()
                    ax.pie(pollutant_totals.values(), labels=pollutant_totals.keys(), autopct="%1.1f%%", startangle=90, 
                           colors=sns.color_palette("Set3", len(pollutant_totals)))
                    ax.set_title(f"Distribution of Pollutants in {country}")
                    st.pyplot(fig)

                # Predict the most polluted country among the selected ones
                st.subheader("Compare the Most Polluted Country Among Selected Countries (Predicted AQI)")

                predicted_aqi_values = []

                # Loop through the selected countries and predict AQI for each city in the country
                for country in selected_countries:
                    country_data = analysis_data[analysis_data['Country'] == country]
                    aqi_values = []

                    for city in country_data['City'].unique():
                        try:
                            aqi_value, _ = predict_aqi_with_defaults(country, city, dataset)
                            aqi_values.append(aqi_value)
                        except ValueError:
                            continue  # Skip cities that cause an error in prediction
                    
                    if aqi_values:  # Check if there are any predictions
                        avg_aqi = sum(aqi_values) / len(aqi_values)  # Average AQI for the country
                        predicted_aqi_values.append((country, avg_aqi))

                # Create a DataFrame to visualize the predicted AQI values
                predicted_aqi_df = pd.DataFrame(predicted_aqi_values, columns=['Country', 'Predicted Average AQI'])

                # Sort the countries by the predicted average AQI in descending order
                predicted_aqi_df = predicted_aqi_df.sort_values(by='Predicted Average AQI', ascending=False)

                # Plot a bar chart for the most polluted countries based on predicted AQI
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=predicted_aqi_df, x='Country', y='Predicted Average AQI', ax=ax, palette='viridis')
                ax.set_title('Top 5 Most Polluted Countries by Predicted Average AQI')
                ax.set_xlabel('Country')
                ax.set_ylabel('Predicted Average AQI Value')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

    elif page == "Visualization":
        st.header("Generate Insights and Visualizations")
        upload_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"], key="visualization")

        if upload_file is not None:
            data = pd.read_csv(upload_file)
            st.success("Dataset loaded successfully!")

            # Interactive Map using Plotly
            st.subheader("1. Geographical Distribution of AQI (Interactive Map)")
            fig = px.scatter_geo(data,
                                 lat='lat',
                                 lon='lng',
                                 color='AQI Value', 
                                 hover_name='City', 
                                 hover_data={'AQI Value': True, 'AQI Category': True},
                                 color_continuous_scale="Viridis", 
                                 title="Geographical Visualization of AQI",
                                 projection="natural earth")
            fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightgray")
            fig.update_layout(
                geo=dict(scope='world'),
                title="AQI Distribution on the World Map"
            )
            st.plotly_chart(fig)

            # Choropleth Map using Plotly
            st.subheader("2. Average AQI by Country (Choropleth Map)")
            avg_aqi_by_country = data.groupby('Country')['AQI Value'].mean().reset_index()
            fig = px.choropleth(avg_aqi_by_country,
                                locations='Country',
                                locationmode='country names',
                                color='AQI Value',
                                hover_name='Country',
                                color_continuous_scale="Viridis",
                                labels={'AQI Value': 'Average AQI'},
                                title="Average AQI by Country")
            fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightgray")
            st.plotly_chart(fig)

            # Visualize AQI Category Distribution
            st.subheader("3. AQI Category Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=data, x='AQI Category', order=data['AQI Category'].value_counts().index, ax=ax)
            ax.set_title('AQI Category Distribution')
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            st.pyplot(fig)

            # Visualize Top 10 Polluted Cities by AQI Value
            st.subheader("4. Top 10 Polluted Cities by AQI Value")
            top_cities_data = data.groupby('City')['AQI Value'].mean().sort_values(ascending=False).head(10).reset_index()
            fig, ax = plt.subplots()
            sns.barplot(data=top_cities_data, x='City', y='AQI Value', ci=None, ax=ax)
            ax.set_title('Top 10 Polluted Cities by AQI Value')
            ax.set_xlabel('City')
            ax.set_ylabel('Average AQI Value')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

            # Additional Plots (e.g., AQI Distribution, Correlation Heatmap)
            st.subheader("5. AQI Values by Top 5 Countries")
            top_5_countries = data.groupby('Country')['AQI Value'].mean().sort_values(ascending=False).head(5)
            fig, ax = plt.subplots(figsize=(12, 6))
            top_5_countries.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title('Average AQI Values for Top 5 Countries')
            ax.set_xlabel('Country')
            ax.set_ylabel('Average AQI Value')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

            # Pollutant Contributions by AQI Category
            st.subheader("6. Pollutant Contributions by AQI Category")
            subset_melted = data.melt(id_vars=['AQI Category'], 
                                      value_vars=['PM2.5 AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value'],
                                      var_name='Pollutant', value_name='Value')
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='AQI Category', y='Value', hue='Pollutant', data=subset_melted, ci=None, palette='pastel', ax=ax)
            ax.set_title('Pollutant Contributions by AQI Category')
            st.pyplot(fig)

            # Correlation Matrix Heatmap
            st.subheader("7. Correlation Matrix")
            corr = data[['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
