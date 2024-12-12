import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)

    def extract_salary_range(salary_str):
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

    data[['Min Salary', 'Max Salary']] = data['Salary'].apply(
        lambda x: pd.Series(extract_salary_range(x))
    )

    data_cleaned = data.dropna(subset=['Min Salary', 'Max Salary'])
    data_cleaned['Avg Salary'] = (data_cleaned['Min Salary'] + data_cleaned['Max Salary']) / 2
    return data_cleaned

def get_salary_stats(data_cleaned):
    return data_cleaned[['Min Salary', 'Max Salary', 'Avg Salary']].describe()

def plot_salary_distribution(data_cleaned):
    plt.figure(figsize=(10, 6))
    sns.histplot(data_cleaned['Avg Salary'], kde=True, bins=30, color='blue')
    plt.title('Salary Distribution')
    plt.xlabel('Average Salary')
    plt.ylabel('Frequency')
    plt.show()

def perform_kmeans(data_cleaned):
    kmeans = KMeans(n_clusters=3, random_state=42)
    data_cleaned['Cluster'] = kmeans.fit_predict(data_cleaned[['Avg Salary']])
    return data_cleaned, kmeans

def perform_regression(data_cleaned):
    regression = LinearRegression()
    data_cleaned['Company Score'] = pd.to_numeric(data_cleaned['Company Score'], errors='coerce')
    data_cleaned.dropna(subset=['Company Score'], inplace=True)
    X = data_cleaned[['Company Score']]
    y = data_cleaned['Avg Salary']
    regression.fit(X, y)
    return regression