import streamlit as st
from Data_Analysis import load_and_clean_data, get_salary_stats, plot_salary_distribution, perform_kmeans, perform_regression
import seaborn as sns
import matplotlib.pyplot as plt

# File path for the dataset
file_path = '/Software Engineer Salaries.csv'

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
st.sidebar.caption("Developed by [Your Name]")

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
    st.markdown("### Data Cleaning Steps")
    st.markdown("- Removed rows with missing or invalid salary data.")
    st.markdown("- Extracted minimum, maximum, and average salary values from salary ranges.")
    
    st.markdown("### Salary Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data_cleaned['Avg Salary'], kde=True, bins=30, color='blue', ax=ax)
    ax.set_title('Salary Distribution', color='white')
    ax.set_xlabel('Average Salary', color='white')
    ax.set_ylabel('Frequency', color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

elif selected_section == "Analysis and Insights":
    st.subheader('Analysis and Insights')
    st.markdown("### Clustering Analysis")
    data_cleaned, kmeans_model = perform_kmeans(data_cleaned)
    cluster_selected = st.selectbox("Select Cluster", options=[0, 1, 2], format_func=lambda x: f"Cluster {x}")
    filtered_data = data_cleaned[data_cleaned['Cluster'] == cluster_selected]
    st.write(f"### Data for Cluster {cluster_selected}")
    st.write(filtered_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=filtered_data['Avg Salary'], y=filtered_data['Company Score'], hue=filtered_data['Cluster'], palette='viridis', ax=ax
    )
    ax.set_title(f'Cluster {cluster_selected}: Salary vs Company Score', color='white')
    ax.set_xlabel('Average Salary', color='white')
    ax.set_ylabel('Company Score', color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    st.markdown("### Regression Analysis")
    regression_model = perform_regression(data_cleaned)
    st.write('Regression Coefficients:', regression_model.coef_)
    st.write('Regression Intercept:', regression_model.intercept_)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(
        x=data_cleaned['Company Score'],
        y=data_cleaned['Avg Salary'],
        scatter_kws={'alpha':0.6},
        line_kws={'color':'red'},
        ax=ax
    )
    ax.set_title('Regression Analysis: Salary vs Company Score', color='white')
    ax.set_xlabel('Company Score', color='white')
    ax.set_ylabel('Average Salary', color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

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