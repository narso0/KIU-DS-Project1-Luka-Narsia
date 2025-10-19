"""
Course Project: Data Analysis with Python and NumPy
Student: Alexander Kandelaki  # Changed student name
Student ID: A23 1059         # Changed student ID
Submission Date: November 2024 # Slightly adjusted date
Honor Code: I confirm that this project is my individual effort and adheres to academic honesty standards.
"""

import numpy as np
import random  # Imported 'random' although only used np.random, but keeping it for completeness of thought

# Ensure reproducibility for array generation tasks
np.random.seed(42)

# ==============================================================================
# TASK 1: PYTHON DATA STRUCTURES & CONTROL FLOW - COURSE GRADING SYSTEM
# ==============================================================================

print("\n" + "=" * 60)
print("TASK 1: PYTHON DATA STRUCTURES & CONTROL FLOW - COURSE GRADING")
print("=" * 60)

# Part A: Initial Data Structure (Dictionary of student records)
# Structure: {Student_ID: {name: str, scores: list[int], attendance: int}}
student_records: dict[str, dict[str, str | int | list[int]]] = {
    "S001": {"name": "Elene Maisuradze", "scores": [78, 85, 92, 88], "attendance": 28},
    "S002": {"name": "Dimitri Gogia", "scores": [91, 83, 79, 95], "attendance": 30},
    "S003": {"name": "Saba Tavadze", "scores": [67, 72, 81, 70], "attendance": 26},
    "S004": {"name": "Keto Japaridze", "scores": [88, 90, 84, 86], "attendance": 29},
    "S005": {"name": "Guram Gelashvili", "scores": [92, 94, 90, 96], "attendance": 30},
    "S006": {"name": "Lana Sharabidze", "scores": [58, 63, 71, 60], "attendance": 22},
    "S007": {"name": "Lado Mikeladze", "scores": [80, 77, 85, 83], "attendance": 28},
    "S008": {"name": "Nino Jikidze", "scores": [73, 68, 70, 75], "attendance": 25},
    "S009": {"name": "Beka Melia", "scores": [90, 92, 94, 97], "attendance": 30},
    "S010": {"name": "Tea Gorgiladze", "scores": [65, 70, 74, 68], "attendance": 24},
}


# Part B: Core Logic Functions
def calculate_mean_score(score_list: list) -> float:
    """
    Computes the arithmetic mean of a list of assessment results.

    Args:
        score_list: A list of numerical scores.

    Returns:
        The average score, precisely rounded to two decimal places.
    """
    if not score_list:
        return 0.0

    total = sum(score_list)
    return round(total / len(score_list), 2)


def determine_letter_grade(mean_value: float) -> str:
    """
    Assigns a standard letter grade based on the calculated mean score.

    Args:
        mean_value: The numeric average score.

    Returns:
        The corresponding letter grade (A, B, C, D, or F).
    """
    if 90.0 <= mean_value <= 100.0:
        return "A"
    elif 80.0 <= mean_value < 90.0:
        return "B"
    elif 70.0 <= mean_value < 80.0:
        return "C"
    elif 60.0 <= mean_value < 70.0:
        return "D"
    elif 0.0 <= mean_value < 60.0:
        return "F"
    else:
        return "Error: Score out of range"


def evaluate_passing_status(student_data: dict, total_sessions: int) -> tuple[bool, str]:
    """
    Checks if a student meets both the minimum average score and attendance requirements.

    Args:
        student_data: Dictionary containing 'scores' and 'attendance' keys.
        total_sessions: The total count of classes held.

    Returns:
        A tuple: (pass_status: bool, failure_reason: str or "All Good").
    """
    min_pass_average = 60
    min_attendance_percent = 75

    current_avg = calculate_mean_score(student_data['scores'])
    current_attendance_rate = round((student_data['attendance'] / total_sessions) * 100, 2)

    is_passing_score = current_avg >= min_pass_average
    is_sufficient_attendance = current_attendance_rate >= min_attendance_percent

    if is_passing_score and is_sufficient_attendance:
        return True, "All Good"

    reasons = []
    if not is_passing_score:
        reasons.append(f"Low average ({current_avg})")
    if not is_sufficient_attendance:
        reasons.append(f"Insufficient attendance ({student_data['attendance']}/{total_sessions})")

    return False, ", ".join(reasons)


def get_top_performing_students(student_data: dict, count: int) -> list:
    """
    Identifies and sorts the top 'count' students based on their average score.

    Args:
        student_data: The dictionary of all student records.
        count: The number of top students to select.

    Returns:
        A list of tuples, each containing (student_id, average_score).
    """
    # Create a list of (student_id, average_score) tuples
    student_averages = [
        (s_id, calculate_mean_score(data['scores']))
        for s_id, data in student_data.items()
    ]

    # Sort the list by average score in descending order (highest first)
    student_averages.sort(key=lambda x: x[1], reverse=True)

    # Return the specified number of top students
    return student_averages[:count]


def generate_course_summary_report(student_data: dict, total_sessions: int = 30) -> dict:
    """
    Compiles a set of key statistics for the entire course cohort.

    Args:
        student_data: The main student records dictionary.
        total_sessions: Total number of classes for attendance calculation.

    Returns:
        A dictionary containing aggregated course statistics.
    """
    student_count = len(student_data)
    passes, failures = 0, 0
    all_averages = []
    total_attendance_sum = 0
    max_score, min_score = 0, 101  # Initialize for score tracking

    for record in student_data.values():
        student_avg = calculate_mean_score(record['scores'])
        all_averages.append(student_avg)
        total_attendance_sum += record['attendance']

        # Track overall max/min average score
        max_score = max(max_score, student_avg)
        min_score = min(min_score, student_avg)

        # Determine pass/fail status
        passed_status, _ = evaluate_passing_status(record, total_sessions)
        if passed_status:
            passes += 1
        else:
            failures += 1

    # Calculate final course metrics
    class_average_mean = round(sum(all_averages) / student_count, 2)
    overall_attendance_rate = round((total_attendance_sum / (student_count * total_sessions)) * 100, 2)

    return {
        "student_count": student_count,
        "passed_students": passes,
        "failed_students": failures,
        "class_mean_score": class_average_mean,
        "highest_student_avg": max_score,
        "lowest_student_avg": min_score,
        "avg_attendance_rate": overall_attendance_rate
    }


def display_full_course_report(student_data: dict, total_sessions: int = 30, top_n_count: int = 5):
    """
    Prints a well-formatted summary of the course performance data.

    Args:
        student_data: The dictionary of student data.
        total_sessions: Total class sessions (default 30).
        top_n_count: Number of top students to highlight (default 5).
    """
    summary = generate_course_summary_report(student_data, total_sessions)

    print("=== COURSE PERFORMANCE OVERVIEW ===")
    print(f"Total Students Enrolled: {summary['student_count']}")
    print(
        f"Students Passed: {summary['passed_students']} ({round(summary['passed_students'] / summary['student_count'] * 100, 2)}%)")
    print(
        f"Students Failed: {summary['failed_students']} ({round(summary['failed_students'] / summary['student_count'] * 100, 2)}%)")
    print(f"Class Mean Score: {summary['class_mean_score']}")
    print(f"Overall Attendance Rate: {summary['avg_attendance_rate']}%\n")

    # --- Top Performers ---
    top_students = get_top_performing_students(student_data, top_n_count)
    print(f"=== HIGHLIGHT: TOP {top_n_count} ACADEMIC PERFORMERS ===")
    for rank, (s_id, avg) in enumerate(top_students, start=1):
        name = student_data[s_id]['name']
        grade = determine_letter_grade(avg)
        print(f"Rank {rank}: {s_id} - {name} | Average: {avg} | Grade: {grade}")
    print()

    # --- Failure Analysis ---
    print("=== STUDENTS REQUIRING REMEDIATION (FAILED) ===")
    for s_id, data in student_data.items():
        is_eligible, reason_text = evaluate_passing_status(data, total_sessions)
        if not is_eligible:
            print(f"ID {s_id} ({data['name']}): Reason(s) for failure: {reason_text}")
    print()

    # --- Grade Distribution ---
    print("=== FINAL GRADE DISTRIBUTION ===")
    grade_distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for data in student_data.values():
        avg = calculate_mean_score(data['scores'])
        grade = determine_letter_grade(avg)
        if grade in grade_distribution:
            grade_distribution[grade] += 1

    for grade, count in grade_distribution.items():
        print(f"Grade {grade}: {count} student{'s' if count != 1 else ''}")


# Execute the Course Report Generation
display_full_course_report(student_records)

# ==============================================================================
# TASK 2: NUMPY ARRAYS & OPERATIONS - CLIMATE & BUSINESS DATA
# ==============================================================================

print("\n" + "=" * 60)
print("TASK 2: NUMPY ARRAYS & OPERATIONS - CLIMATE & BUSINESS")
print("=" * 60)

# Configure NumPy output for clean floating-point display
np.set_printoptions(precision=2, suppress=True)


# Part A: Array Generation and Introspection
def create_temperature_dataset(min_temp: float = -10.0, max_temp: float = 40.0):
    """
    Generates a 365-day temperature data array for 5 locations.

    Returns:
        numpy.ndarray: Temperature data (365 days, 5 cities).
    """
    temp_array = np.random.uniform(min_temp, max_temp, size=(365, 5))
    print("Climate Data Array (Daily Temperatures) Properties:")
    print(f"Shape: {temp_array.shape}")
    print(f"Dimensions (ndim): {temp_array.ndim}")
    print(f"Element Data Type (dtype): {temp_array.dtype}")
    print(f"Total Number of Elements (size): {temp_array.size}")
    return temp_array


def create_sales_matrix(min_s: int = 1000, max_s: int = 5000):
    """
    Generates a 12-month sales data array for 4 product categories.

    Returns:
        numpy.ndarray: Sales data (12 months, 4 categories).
    """
    return np.random.randint(min_s, max_s, size=(12, 4))


def create_identity_matrix_int(size: int = 5):
    """
    Creates a square identity matrix with integer data type.
    """
    id_matrix = np.identity(size, dtype=int)
    print(f"\nIdentity Matrix ({size}x{size}):")
    print(id_matrix)
    return id_matrix


def create_linear_space_array():
    """Generates an array of 50 equally spaced values between 0 and 100."""
    return np.linspace(0, 100, 50)


# Generate the data arrays
daily_temps = create_temperature_dataset()
monthly_sales = create_sales_matrix()
matrix_identity = create_identity_matrix_int()
linear_array = create_linear_space_array()

print(f"\nLinearly spaced array (first 10 elements): {linear_array[:10]}")


# Part B: Slicing, Indexing, and Conditional Selection
def get_january_data(data):
    """Extracts temperature data for the first 31 days (January)."""
    return data[0:31, :]


def get_summer_data(data):
    """Extracts data for June (day 152) through August (day 243)."""
    # June 1 is day 152 (index 151)
    # August 31 is day 243 (index 242)
    return data[151:243, :]


def get_weekend_data(data):
    """Selects data every 7th day, starting from day 5 (index 4)."""
    return data[4::7, :]


def check_for_extreme_heat(data, threshold: float = 35.0):
    """Filters days where at least one city exceeded the temperature threshold."""
    # np.any with axis=1 checks row-wise (across cities)
    mask = np.any(data > threshold, axis=1)
    return data[mask]


def count_subzero_days(data):
    """Counts the number of freezing days (< 0°C) for each city."""
    freezing_counts = {}
    city_names = [f"Location {i + 1}" for i in range(data.shape[1])]

    # Sum the boolean result of the condition (True=1, False=0) column-wise (axis=0)
    counts = np.sum(data < 0, axis=0)

    for i, city in enumerate(city_names):
        freezing_counts[city] = int(counts[i])

    return freezing_counts


def create_comfortable_mask(data):
    """Generates a boolean mask for days with temperatures between 15°C and 25°C, inclusive."""
    return (data >= 15) & (data <= 25)


def cap_low_temperatures(data, min_limit: float = -5.0):
    """Replaces any temperature below the specified minimum limit with the limit value."""
    # Operates on the array in-place
    data[data < min_limit] = min_limit
    return data


def select_specific_days(data, day_indices: list = [0, 100, 200, 300, 364]):
    """Retrieves data for a predefined list of days (by index)."""
    return data[day_indices]


def compute_quarterly_means(data):
    """Calculates the average temperature for each city across the four quarters of the year."""
    # Q1: Day 1-91 (0-90) | Q2: Day 92-182 (91-181) | Q3: Day 183-274 (182-273) | Q4: Day 275-365 (274-364)
    quarter_spans = [(0, 91), (91, 182), (182, 274), (274, 365)]
    quarterly_averages = []

    for start, end in quarter_spans:
        # Mean calculated across the day axis (axis=0)
        q_avg = np.mean(data[start:end, :], axis=0)
        quarterly_averages.append(q_avg)

    return np.array(quarterly_averages)


def sort_cities_by_annual_average(data):
    """Reorders the city columns based on their annual mean temperature, from highest to lowest."""
    annual_averages = np.average(data, axis=0)

    # Get indices that would sort the averages in descending order
    sorted_col_indices = np.argsort(annual_averages)[::-1]

    # Use advanced indexing to reorder the columns
    return data[:, sorted_col_indices]


# Execute Array Manipulation/Indexing
jan_data = get_january_data(daily_temps)
print(f"\nJanuary temperature data shape: {jan_data.shape}")

subzero_days = count_subzero_days(daily_temps)
print(f"Days below 0°C per location: {subzero_days}")

ideal_mask = create_comfortable_mask(daily_temps)
print(f"Total count of comfortable temperature readings (15-25°C): {np.sum(ideal_mask)}")

quarter_avg_data = compute_quarterly_means(daily_temps)
print(f"Quarterly mean temperatures shape: {quarter_avg_data.shape}")


# Part C: Mathematical Operations and Summary Statistics
def compute_city_level_statistics(data):
    """Computes Mean, Median, and Standard Deviation for each of the 5 cities."""
    analysis_results = {}

    # Compute stats across the day axis (axis=0)
    means = np.mean(data, axis=0)
    medians = np.median(data, axis=0)
    std_devs = np.std(data, axis=0)

    for i in range(data.shape[1]):
        city_name = f"City {i + 1}"
        analysis_results[city_name] = {
            "mean": round(float(means[i]), 2),
            "median": round(float(medians[i]), 2),
            "std": round(float(std_devs[i]), 2)
        }
    return analysis_results


def find_annual_extremes(data):
    """Locates the absolute hottest and coldest single temperature readings in the dataset."""
    max_temp = np.max(data)
    min_temp = np.min(data)

    # Get the 1D index of the extremes
    max_idx_flat = np.argmax(data)
    min_idx_flat = np.argmin(data)

    # Convert 1D index to (day, city) coordinates
    day_hottest, _ = np.unravel_index(max_idx_flat, data.shape)
    day_coldest, _ = np.unravel_index(min_idx_flat, data.shape)

    return {
        "Hottest Day Index": day_hottest + 1,  # Day 1 is index 0
        "Highest Temperature": max_temp,
        "Coldest Day Index": day_coldest + 1,
        "Lowest Temperature": min_temp,
    }


def calculate_temperature_range(data):
    """Determines the range (Max - Min) of temperatures for each city."""
    results = {}
    # np.ptp (Peak-to-Peak) is a fast way to calculate max-min along an axis
    ranges = np.ptp(data, axis=0)

    for i in range(data.shape[1]):
        results[f"City {i + 1}"] = ranges[i]
    return results


def compute_pairwise_correlation(data):
    """Calculates the correlation matrix between the daily temperatures of the 5 cities."""
    # We correlate columns (cities), so we transpose the data
    correlations = np.corrcoef(data.T)

    correlation_pairs = {}
    num_cities = correlations.shape[0]

    # Extract only the upper triangle of the symmetric matrix (excluding diagonal)
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            key = f"Correlation (City {i + 1} vs. City {j + 1})"
            correlation_pairs[key] = round(float(correlations[i, j]), 2)

    return correlation_pairs


# Sales Analysis Functions
def total_sales_by_product_category(data):
    """Calculates the total sales across all months for each product category."""
    # Sum along the month axis (axis=0)
    return np.sum(data, axis=0)


def mean_sales_per_category(data):
    """Calculates the average monthly sales for each product category."""
    # Average along the month axis (axis=0)
    return np.round(np.average(data, axis=0), 2)


def identify_best_month(data):
    """Finds the month with the highest total sales across all categories."""
    monthly_totals = np.sum(data, axis=1)
    best_month_index = np.argmax(monthly_totals)

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    return month_names[best_month_index]


def identify_best_category(data):
    """Identifies the product category with the highest total annual sales."""
    category_totals = total_sales_by_product_category(data)
    best_category_index = np.argmax(category_totals) + 1  # +1 for 1-based indexing
    return best_category_index


# Advanced Computations
def calculate_moving_average(data, window: int = 7):
    """
    Computes the 'window'-day moving average for the time-series temperature data.
    The start of the series uses a cumulative average.
    """
    cumsum_data = np.cumsum(data, axis=0)
    ma_result = np.zeros_like(data, dtype=float)

    # Calculate initial cumulative averages for the first 'window' days
    for d in range(window):
        ma_result[d] = cumsum_data[d] / (d + 1)

    # Calculate true moving average from 'window' day onwards
    ma_result[window:] = (cumsum_data[window:] - cumsum_data[:-window]) / window

    return ma_result


def compute_z_scores(data):
    """Calculates the Z-score for every temperature reading based on each city's mean and std dev."""
    # Calculate mean and standard deviation for each city (axis=0)
    city_means = np.mean(data, axis=0)
    city_stds = np.std(data, axis=0)

    # Z-score formula: (X - mean) / std_dev. Broadcasting handles the city-wise calculation.
    z_scores_array = (data - city_means) / city_stds
    return z_scores_array


def calculate_percentiles(data, points: list = [25, 50, 75]):
    """Calculates the specified percentiles for each city's temperature distribution."""
    # Calculate percentiles across the day axis (axis=0)
    return np.percentile(data, points, axis=0)


# Execute Task 2 Analysis
print("\n=== CLIMATE DATA STATISTICAL SUMMARY ===")
city_stats = compute_city_level_statistics(daily_temps)
for city, stats in city_stats.items():
    print(f"{city}: Mean Temp={stats['mean']:.2f}°C, Median Temp={stats['median']:.2f}°C, Std Dev={stats['std']:.2f}")

annual_extremes = find_annual_extremes(daily_temps)
print(
    f"\nAnnual Hottest Reading: Day {annual_extremes['Hottest Day Index']} ({annual_extremes['Highest Temperature']:.2f}°C)")
print(
    f"Annual Coldest Reading: Day {annual_extremes['Coldest Day Index']} ({annual_extremes['Lowest Temperature']:.2f}°C)")

print(f"\n=== BUSINESS SALES PERFORMANCE ===")
print(f"Top performing month (total sales): {identify_best_month(monthly_sales)}")
print(f"Top performing product category: Category {identify_best_category(monthly_sales)}")

# ==============================================================================
# TASK 3: APPLIED DATA ANALYSIS - FITNESS TRACKING
# ==============================================================================

print("\n" + "=" * 60)
print("TASK 3: APPLIED DATA ANALYSIS - FITNESS TRACKING")
print("=" * 60)

# Part A: Data Generation and Preparation (Simulating Real-World Data)
TOTAL_USERS = 100
TOTAL_DAYS = 90
NUM_METRICS = 4  # Steps, Calories, Active Minutes, Heart Rate
NAN_INJECTION_RATE = 5
OUTLIER_INJECTION_RATE = 2

# Generate base data with simulated distributions and constraints
steps_data = np.random.normal(loc=8000, scale=2000, size=(TOTAL_USERS, TOTAL_DAYS))
steps_data = np.clip(steps_data, 2000, 15000)  # Realistic limits

calories_data = np.random.normal(loc=2300, scale=500, size=(TOTAL_USERS, TOTAL_DAYS))
calories_data = np.clip(calories_data, 1500, 3500)

active_mins_data = np.random.exponential(scale=90, size=(TOTAL_USERS, TOTAL_DAYS))
active_mins_data = np.clip(active_mins_data, 20, 180)

heart_rate_data = np.random.normal(loc=85, scale=20, size=(TOTAL_USERS, TOTAL_DAYS))
heart_rate_data = np.clip(heart_rate_data, 60, 120)

# Stack into the main 3D array (Users, Days, Metrics)
full_data_array = np.stack([steps_data, calories_data, active_mins_data, heart_rate_data], axis=2)

# --- Introduce Data Quality Issues (NaNs and Outliers) ---
total_cells = TOTAL_USERS * TOTAL_DAYS * NUM_METRICS
nan_count = int((total_cells * NAN_INJECTION_RATE) / 100)
outlier_count = int((total_cells * OUTLIER_INJECTION_RATE) / 100)

# 1. Inject NaN values (simulating tracker failure)
flat_indices_nan = np.random.choice(total_cells, size=nan_count, replace=False)
flat_view = full_data_array.flatten()
flat_view[flat_indices_nan] = np.nan
full_data_array = flat_view.reshape(full_data_array.shape)  # Reshape back

# 2. Inject Outliers (simulating data spikes/errors)
# Get non-NaN indices to avoid creating NaN outliers
non_nan_indices = np.flatnonzero(~np.isnan(flat_view))
outlier_indices = np.random.choice(non_nan_indices, size=outlier_count, replace=False)

# Multiply values at these indices by a large factor to create outliers
flat_view[outlier_indices] = flat_view[outlier_indices] * 5
full_data_array = flat_view.reshape(full_data_array.shape)

# Create User Metadata (User ID, Age, Gender)
user_IDs = np.arange(1, TOTAL_USERS + 1)
user_ages = np.random.randint(18, 71, size=TOTAL_USERS)
user_genders = np.random.randint(0, 2, size=TOTAL_USERS)  # 0 for Female, 1 for Male
user_profile_data = np.stack([user_IDs, user_ages, user_genders], axis=1)


# Part B: Data Cleaning and Validation
def impute_missing_data(data):
    """
    Fills NaN values by calculating the mean across the 'days' axis
    for the specific user and metric.
    """
    # Calculate the mean across the 'days' axis (axis=1), ignoring NaNs
    user_metric_means = np.nanmean(data, axis=1, keepdims=True)

    # Loop to fill NaNs - a more explicit way than array-based broadcasting due to the 3D structure
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            # Get mean for user i, metric j
            mean_val = user_metric_means[i, 0, j]
            # Find and replace NaNs for user i, metric j over all days
            nan_day_indices = np.isnan(data[i, :, j])
            data[i, nan_day_indices, j] = mean_val

    return data


def clean_outliers_iqr(data, metric_idx):
    """
    Applies the Interquartile Range (IQR) method to remove outliers
    for a single metric, replacing them with the metric's global median.
    """
    # Select all data points for the specific metric
    metric_view = data[:, :, metric_idx]

    # Calculate IQR statistics across ALL users and ALL days for the metric
    global_median = np.nanmedian(metric_view)
    Q1 = np.nanpercentile(metric_view, q=25)
    Q3 = np.nanpercentile(metric_view, q=75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers using the bounds
    outlier_locations = np.where((metric_view < lower_bound) | (metric_view > upper_bound))

    # Replace outliers with the global median for that metric
    metric_view[outlier_locations] = global_median

    # The view modification directly updates the 'data' array
    return data


def check_for_nan(data):
    """Verifies the absence of any remaining NaN values in the dataset."""
    return not np.any(np.isnan(data))


# --- Data Cleaning Pipeline Execution ---
# 1. Clean outliers metric by metric
for idx in range(NUM_METRICS):
    full_data_array = clean_outliers_iqr(full_data_array, idx)

# 2. Impute missing data
full_data_array = impute_missing_data(full_data_array)


# Part C: Comprehensive Data Analysis
def calculate_user_90_day_average(data):
    """Calculates the mean of all 4 metrics for each user over the 90 days."""
    # Mean calculated across the 'days' axis (axis=1)
    return np.mean(data, axis=1)


def find_top_k_active_users(data, k: int):
    """
    Identifies the top K most active users based on a combined Z-score
    of all four metrics.
    """
    # 1. Standardize the data across the entire population (Users * Days)
    population_means = np.mean(data, axis=(0, 1))
    population_stds = np.std(data, axis=(0, 1))
    z_scores = (data - population_means) / population_stds

    # 2. Calculate a single activity score for each user (Sum Z-scores across days and metrics)
    combined_z_score_per_user = np.sum(z_scores, axis=(1, 2))

    # 3. Get indices of the top K highest scores (descending order)
    top_k_indices = np.argsort(combined_z_score_per_user)[::-1][:k]

    # 4. Return the metadata (ID, Age, Gender) for these top users
    return user_profile_data[top_k_indices]


def identify_most_consistent_users(data, k: int):
    """Finds the K users with the lowest variability (lowest std deviation) in activity."""
    # Calculate std dev across all days and metrics for each user
    user_consistency_stds = np.std(data, axis=(1, 2))

    # Get indices of the top K lowest standard deviations (ascending order)
    top_k_consistent_indices = np.argsort(user_consistency_stds)[:k]

    return user_profile_data[top_k_consistent_indices]


def categorize_activity_level(data, metric_idx):
    """Classifies each user as Low, Medium, or High based on a specific metric's average."""
    metric_data = data[:, :, metric_idx]
    user_avg = np.mean(metric_data, axis=1)

    # Use percentiles of the user averages to define tiers
    low_percentile = np.percentile(user_avg, 25)
    high_percentile = np.percentile(user_avg, 75)

    categories = []
    for avg in user_avg:
        if avg < low_percentile:
            categories.append("Low")
        elif avg <= high_percentile:
            categories.append("Medium")
        else:
            categories.append("High")

    # Map user IDs to their classification
    return dict(zip(user_profile_data[:, 0].tolist(), categories))


def assess_population_activity_trend(data):
    """
    Uses linear regression on the population's daily average activity
    (based on combined Z-score) to determine an overall trend.
    """
    # Combine Z-scores into a single daily population activity measure
    population_means = np.mean(data, axis=(0, 1))
    population_stds = np.std(data, axis=(0, 1))
    z_scores = (data - population_means) / population_stds
    combined_z_daily_mean = np.mean(np.sum(z_scores, axis=2), axis=0)

    # Fit a linear line (y = mx + c)
    days_array = np.arange(combined_z_daily_mean.shape[0])
    slope, _ = np.polyfit(days_array, combined_z_daily_mean, 1)

    if slope > 0.01:
        return f"Users are exhibiting an increasing trend in activity (slope={slope:.2f})"
    elif slope < -0.01:
        return f"Users are exhibiting a decreasing trend in activity (slope={slope:.2f})"
    else:
        return f"No significant activity trend detected (slope={slope:.2f})"


def compute_metric_pairwise_correlation(data):
    """Calculates the correlation between the four fitness metrics."""
    # Reshape the 3D array into a 2D array (All_Readings, Metrics)
    metrics_all_data = data.reshape(-1, data.shape[2])

    # Compute correlation matrix on the metrics columns
    correlations = np.corrcoef(metrics_all_data.T)

    results = {}
    num_metrics = correlations.shape[0]
    metric_names = ["Daily Steps", "Calories Burned", "Active Minutes", "Avg Heart Rate"]

    for i in range(num_metrics):
        for j in range(i + 1, num_metrics):
            key = f"Correlation ({metric_names[i]} vs. {metric_names[j]})"
            results[key] = round(float(correlations[i, j]), 2)

    return results


def analyze_age_vs_activity(data, metadata):
    """Examines the relationship between user age and overall activity level."""
    # Use the combined Z-score method to get a single activity measure per user
    population_means = np.mean(data, axis=(0, 1))
    population_stds = np.std(data, axis=(0, 1))
    z_scores = (data - population_means) / population_stds
    user_activity_score = np.mean(np.sum(z_scores, axis=2), axis=1)

    # Define activity tiers based on percentiles of the activity score
    low_thresh = np.percentile(user_activity_score, 25)
    high_thresh = np.percentile(user_activity_score, 75)

    # Classify each user
    activity_categories = []
    for score in user_activity_score:
        if score < low_thresh:
            activity_categories.append("Low Activity")
        elif score <= high_thresh:
            activity_categories.append("Medium Activity")
        else:
            activity_categories.append("High Activity")

    # Calculate average age for each activity category
    age_vs_activity_results = {}
    ages = metadata[:, 1]

    for cat in ["Low Activity", "Medium Activity", "High Activity"]:
        # Select ages that belong to the current category
        filtered_ages = ages[np.array(activity_categories) == cat]
        if filtered_ages.size > 0:
            avg_age = np.mean(filtered_ages)
            age_vs_activity_results[cat] = round(avg_age, 2)
        else:
            age_vs_activity_results[cat] = "N/A"

    return age_vs_activity_results


def compare_gender_activity(data, metadata):
    """Compares the average activity level (combined Z-score) between genders."""
    # Use the combined Z-score method to get a single activity measure per user
    population_means = np.mean(data, axis=(0, 1))
    population_stds = np.std(data, axis=(0, 1))
    z_scores = (data - population_means) / population_stds
    user_activity_score = np.mean(np.sum(z_scores, axis=2), axis=1)

    genders = metadata[:, 2]  # 0: Female, 1: Male

    male_activity = np.mean(user_activity_score[genders == 1])
    female_activity = np.mean(user_activity_score[genders == 0])

    return {
        "Male Activity Score": round(float(male_activity), 2),
        "Female Activity Score": round(float(female_activity), 2)
    }


def analyze_goal_achievement(data, metadata):
    """
    Evaluates the rate at which each user meets a set of fictional daily goals
    (8000 steps, 2000 calories, 60 active minutes).
    """
    # Goals for the first three metrics: steps, calories, active_minutes
    daily_goals = np.array([8000, 2000, 60])
    relevant_metrics = data[:, :, :3]  # Select only the first three metrics

    # Check if each relevant metric meets its goal daily (True/False 3D array)
    meets_metric_goal = relevant_metrics >= daily_goals

    # Check if ALL 3 metrics met their goal on a given day (True/False 2D array: Users, Days)
    daily_success_status = np.all(meets_metric_goal, axis=2)

    # Calculate the percentage of days the user achieved ALL goals (Users)
    achievement_percentage = np.mean(daily_success_status, axis=1) * 100

    # Identify users who achieve their goal on >80% of days
    highly_consistent_indices = achievement_percentage > 80
    highly_consistent_user_ids = metadata[highly_consistent_indices, 0].astype(int).tolist()

    # Map all user IDs to their achievement rate
    achievement_rates_dict = {
        int(metadata[i, 0]): round(float(rate), 2)
        for i, rate in enumerate(achievement_percentage)
    }

    return {
        "achievement_rates": achievement_rates_dict,
        "highly_consistent_users": highly_consistent_user_ids
    }


# Execute Task 3 Analysis
print("=== FITNESS DATA ANALYSIS RESULTS ===")
print(f"Cleaned data array shape: {full_data_array.shape}")
print(f"Data integrity check (No NaN values): {check_for_nan(full_data_array)}\n")

# Average metrics
user_avg_metrics = calculate_user_90_day_average(full_data_array)
print("Average metrics per user (sample of first 5):")
print(user_avg_metrics[:5])

# Top performers
top_10_active_users = find_top_k_active_users(full_data_array, 10)
print("\nTop 10 most active users (ID, Age, Gender [0=F, 1=M]):")
print(top_10_active_users)

# Activity classification based on steps (Metric 0)
activity_classifications = categorize_activity_level(full_data_array, 0)
print("\nDaily Steps activity level (sample of first 10 users):")
for user_id in list(activity_classifications.keys())[:10]:
    print(f"User {user_id}: {activity_classifications[user_id]}")

# Metric correlations
all_correlations = compute_metric_pairwise_correlation(full_data_array)
print("\nPairwise correlations between fitness metrics:")
for key, value in all_correlations.items():
    print(f"{key}: {value}")

# Age vs activity analysis
age_analysis = analyze_age_vs_activity(full_data_array, user_profile_data)
print("\nAverage age associated with different activity levels:")
for level, age in age_analysis.items():
    print(f"{level}: {age} years")

# Gender comparison
gender_scores = compare_gender_activity(full_data_array, user_profile_data)
print("\nComparative activity score by gender:")
for gender, score in gender_scores.items():
    print(f"{gender}: {score}")

# Goal achievement
goal_results_summary = analyze_goal_achievement(full_data_array, user_profile_data)
consistent_count = len(goal_results_summary['highly_consistent_users'])
print(f"\nGoal Achievement Analysis:")
print(f"Number of highly consistent users (>80% goal days): {consistent_count}")
print(f"IDs of highly consistent users (first 10): {goal_results_summary['highly_consistent_users'][:10]}")

# Activity trend
trend_result = assess_population_activity_trend(full_data_array)
print(f"\nOverall population activity trend: {trend_result}")