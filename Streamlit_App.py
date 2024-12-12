import streamlit as st
from Data_Analysis import load_and_clean_data, get_salary_stats, plot_salary_distribution, perform_kmeans, perform_regression
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

current_dir = Path(__file__).parent
# File path for the dataset
file_path = current_dir / "Software Engineer Salaries.csv"

# Load and clean data
data_cleaned = load_and_clean_data(file_path)

# Streamlit App Configuration
st.set_page_config(
    page_title="Software Engineer Salaries Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("📊 Software Engineer Salaries")
st.sidebar.markdown("---")
st.sidebar.header("Navigation")
sections = [
    "Overview",
    "Data Exploration and Preparation",
    "Analysis and Insights",
    "Conclusions and Recommendations"
]
selected_section = st.sidebar.radio("Choose a Section:", sections)

# Add extra information to the sidebar
st.sidebar.markdown("---")
st.sidebar.header("About the App")
st.sidebar.info(
    "This app analyzes software engineer salaries, exploring patterns and trends using clustering "
    "and regression analysis. The interactive visualizations provide insights for both job seekers "
    "and employers."
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Pardillo Boys")

# Styling
st.markdown(
    """
    <style>
    .main {background-color: #1E1E1E; color: white;}
    .stSidebar {background-color: #2C2F33;}
    h1, h2, h3, h4 {color: #FFD700;}
    </style>
    """,
    unsafe_allow_html=True
)

# Display Sections
if selected_section == "Overview":
    st.title('Software Engineer Salaries Analysis')
    st.subheader('Overview')
    st.markdown(
        """
        This application explores a dataset of software engineer salaries. The goal is to identify patterns
        and trends in salaries and their relationship with company scores. We use techniques such as clustering
        and regression analysis to derive insights.
        """
    )
    st.write("### Dataset Sample")
    st.write(data_cleaned.head())

elif selected_section == "Data Exploration and Preparation":
    st.subheader('Data Exploration and Preparation')
    
    # Data Cleaning Steps
    st.markdown("### Data Cleaning Steps")
    st.markdown("- **Removed rows with missing or invalid salary data.**")
    st.markdown("- **Extracted minimum, maximum, and average salary values from salary ranges.**")
    
    # Salary Distribution Plot
    st.markdown("### Salary Distribution")

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot histogram of Average Salary using seaborn
    sns.histplot(
        data=data_cleaned,
        x='Avg Salary',
        kde=True,  # Include kernel density estimate
        bins=30,
        color='skyblue',
        edgecolor='black',
        ax=ax
    )

    # Add labels and title
    ax.set_title('Distribution of Average Salaries for Software Engineers', fontsize=16)
    ax.set_xlabel('Average Salary (USD)', fontsize=12)
    ax.set_ylabel('Number of Job Listings', fontsize=12)

    # Customize tick parameters and grid
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Add interpretation and insights
    st.markdown(
        """
        **Interpretation:**

        - The histogram illustrates the distribution of average salaries for software engineering positions.
        - The x-axis represents the average salary in USD.
        - The y-axis shows the number of job listings corresponding to each salary range.
        - The overlaid KDE (Kernel Density Estimate) curve provides a smoothed estimate of the salary distribution.
        - Peaks in the histogram indicate salary ranges with a higher concentration of job listings.
        """
    )


elif selected_section == "Analysis and Insights":
    st.subheader('Analysis and Insights')
    
    # Clustering Analysis Section
    st.markdown("### Clustering Analysis")
    st.markdown(
        """
        Clustering analysis is performed to group companies based on the average salaries they offer.
        This helps identify patterns in salary ranges and company types.
        """
    )

    # Perform K-Means clustering
    data_cleaned, kmeans_model = perform_kmeans(data_cleaned)
    
    # Dropdown to select a cluster
    cluster_selected = st.selectbox(
        "Select a Cluster to Explore:", 
        options=[0, 1, 2], 
        format_func=lambda x: f"Cluster {x}"
    )

    # Filter data for the selected cluster
    filtered_data = data_cleaned[data_cleaned['Cluster'] == cluster_selected]
    
    # Display filtered data for the selected cluster
    st.write(f"### Data for Cluster {cluster_selected}")
    st.dataframe(filtered_data[['Company', 'Job Title', 'Avg Salary', 'Company Score', 'Location']])

    # Visualization: Scatter plot for selected cluster
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=filtered_data,
        x='Avg Salary',
        y='Company Score',
        hue='Cluster',
        palette='viridis',
        ax=ax,
        s=100
    )
    ax.set_title(f'Cluster {cluster_selected}: Salary vs Company Score', fontsize=16)
    ax.set_xlabel('Average Salary (USD)', fontsize=12)
    ax.set_ylabel('Company Score', fontsize=12)
    ax.grid(axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    st.pyplot(fig)

    # Explanation of clustering insights
    st.markdown(
        f"""
        **Cluster {cluster_selected} Insights:**
        - The scatter plot shows the relationship between company scores and average salaries for this cluster.
        - Companies in this cluster offer average salaries typically in the range of **${filtered_data['Avg Salary'].min():,.0f}** to **${filtered_data['Avg Salary'].max():,.0f}**.
        - This cluster can represent companies with similar compensation structures.
        """
    )
    
    # Regression Analysis Section
    st.markdown("### Regression Analysis")
    st.markdown(
        """
        Regression analysis explores the relationship between company scores and average salaries. 
        A positive slope in the regression line indicates that companies with higher scores 
        tend to offer higher average salaries.
        """
    )

    # Perform regression analysis
    regression_model = perform_regression(data_cleaned)

    # Display regression coefficients
    st.write('### Regression Model Details:')
    st.write(f"- **Coefficient (Impact of Company Score):** {regression_model.coef_[0]:.2f}")
    st.write(f"- **Intercept:** {regression_model.intercept_:.2f}")

    # Visualization: Regression plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.regplot(
        data=data_cleaned,
        x='Company Score',
        y='Avg Salary',
        scatter_kws={'alpha': 0.6},
        line_kws={'color': 'red'},
        ax=ax
    )
    ax.set_title('Regression Analysis: Salary vs Company Score', fontsize=16)
    ax.set_xlabel('Company Score', fontsize=12)
    ax.set_ylabel('Average Salary (USD)', fontsize=12)
    ax.grid(axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    st.pyplot(fig)

    # Explanation of regression insights
    st.markdown(
        """
        **Regression Analysis Insights:**
        - The regression line represents the trend between company scores and average salaries.
        - A positive slope suggests that higher company scores are generally associated with higher salaries.
        - Outliers may indicate companies offering salaries that deviate from the trend, potentially due to location or company size.
        """
    )

elif selected_section == "Conclusions and Recommendations":
    st.subheader('Conclusions and Recommendations')
    st.markdown("### Key Takeaways")
    st.markdown("- **Cluster 1** represents companies offering higher average salaries.")
    st.markdown("- A moderate positive relationship exists between company score and average salary.")
    st.markdown("- Significant variation exists in salary ranges across clusters.")

    st.markdown("### Recommendations")
    st.markdown("- **Job Seekers**: Target companies in Cluster 1 for higher salary opportunities.")
    st.markdown("- **Employers**: Enhance company scores to attract top talent.")
    st.markdown("- Further analyze variations within clusters for more insights.")