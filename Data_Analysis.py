import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Function to load and clean the dataset
def load_and_clean_data(file_path):
    """
    Load the dataset from the given file path, clean the data,
    and extract salary-related information.

    Args:
    - file_path (str): Path to the dataset CSV file.

    Returns:
    - pd.DataFrame: Cleaned dataset with salary range and average salary columns.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Function to extract minimum and maximum salary from salary strings
    def extract_salary_range(salary_str):
        """
        Extract the salary range (min and max) from a salary string.

        Args:
        - salary_str (str): Salary string in the format "$XK - $YK per year".

        Returns:
        - tuple: Minimum salary and maximum salary as integers.
        """
        if isinstance(salary_str, str):
            salary_str = salary_str.replace("$", "").replace("K", "").replace(",", "")
            salary_range = salary_str.split(" - ")
            if len(salary_range) == 2:
                try:
                    min_salary = float(salary_range[0])
                    max_salary = float(salary_range[1].split()[0])
                    return min_salary * 1000, max_salary * 1000
                except ValueError:
                    return None, None
        return None, None

    # Apply the salary extraction function to the Salary column
    data[['Min Salary', 'Max Salary']] = data['Salary'].apply(
        lambda x: pd.Series(extract_salary_range(x))
    )

    # Drop rows with missing salary data
    data_cleaned = data.dropna(subset=['Min Salary', 'Max Salary'])

    # Calculate the average salary for each row
    data_cleaned['Avg Salary'] = (data_cleaned['Min Salary'] + data_cleaned['Max Salary']) / 2

    return data_cleaned

# Function to get salary statistics
def get_salary_stats(data_cleaned):
    """
    Compute descriptive statistics for salary-related columns.

    Args:
    - data_cleaned (pd.DataFrame): Cleaned dataset with salary information.

    Returns:
    - pd.DataFrame: Descriptive statistics for min, max, and average salary.
    """
    return data_cleaned[['Min Salary', 'Max Salary', 'Avg Salary']].describe()

# Function to plot salary distribution
def plot_salary_distribution(data_cleaned):
    """
    Plot the distribution of average salaries in the dataset.

    Args:
    - data_cleaned (pd.DataFrame): Cleaned dataset with salary information.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data_cleaned['Avg Salary'], kde=True, bins=30, color='blue')
    plt.title('Salary Distribution')
    plt.xlabel('Average Salary')
    plt.ylabel('Frequency')
    plt.show()

# Function to perform K-Means clustering on average salary
def perform_kmeans(data_cleaned):
    """
    Perform K-Means clustering on average salaries.

    Args:
    - data_cleaned (pd.DataFrame): Cleaned dataset with salary information.

    Returns:
    - pd.DataFrame: Dataset with assigned cluster labels.
    - KMeans: Trained KMeans model.
    """
    kmeans = KMeans(n_clusters=3, random_state=42)
    data_cleaned['Cluster'] = kmeans.fit_predict(data_cleaned[['Avg Salary']])
    return data_cleaned, kmeans

# Function to perform linear regression on company score and average salary
def perform_regression(data_cleaned):
    """
    Perform linear regression to predict average salary based on company score.

    Args:
    - data_cleaned (pd.DataFrame): Cleaned dataset with salary and company score.

    Returns:
    - LinearRegression: Trained regression model.
    """
    regression = LinearRegression()

    # Ensure the Company Score column is numeric
    data_cleaned['Company Score'] = pd.to_numeric(data_cleaned['Company Score'], errors='coerce')

    # Drop rows with missing company score
    data_cleaned.dropna(subset=['Company Score'], inplace=True)

    # Features (Company Score) and target (Average Salary)
    X = data_cleaned[['Company Score']]
    y = data_cleaned['Avg Salary']

    # Fit the linear regression model
    regression.fit(X, y)

    return regression
