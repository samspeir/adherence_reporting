import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from matplotlib.dates import DateFormatter


# MAKE YOUR SELECTIONS HERE
selected_org = "Clay Health Organization"
start_date = datetime.strptime('2024-01-01', '%Y-%m-%d')
end_date = datetime.now()

# Set up the database connection using SQLAlchemy
def get_database_engine():
    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

    # Replace 'your_region' with the actual AWS region, e.g., 'us-west-2'
    engine = create_engine('awsathena+rest://@athena.us-west-2.amazonaws.com:443/',
                           connect_args={
                               'aws_access_key_id': aws_access_key_id,
                               'aws_secret_access_key': aws_secret_access_key,
                               's3_staging_dir': 's3://ovalspillbucket/',
                               'region_name': 'us-west-1',  
                               'catalog_name': 'athena_oval_data'
                           })
    return engine

def cached_query(query):
    engine = get_database_engine()  # Get or create the engine inside the function to ensure it's fresh
    df = pd.read_sql(query, engine)
    return df


# Now using the cached_query function to execute and cache your queries
user_df = cached_query("""
SELECT id as student_id, name, org_name FROM oval_athena.users
""")

report_df = cached_query("""
    SELECT * FROM (
        SELECT
            student_id,
            test_date,
            appointment_id,
            ROW_NUMBER() OVER (PARTITION BY student_id, test_date ORDER BY test_date) as rn
        FROM oval_athena.report_test_data
    ) sub
    WHERE sub.rn = 1;
""")

def get_unique_org_names():
    engine = get_database_engine()
    query = "SELECT DISTINCT org_name FROM oval_athena.users"
    df = pd.read_sql(query, engine)
    return df['org_name'].tolist()

# Use the function to fetch unique organization names
org_names = get_unique_org_names()

org_names = [org_name if org_name is not None else 'Unknown' for org_name in org_names]

org_names = [str(org_name) for org_name in org_names if org_name is not None]


# Assuming 'test_date' is the column to determine earliest and latest records
# Convert 'test_date' to datetime if not already
report_df['test_date'] = pd.to_datetime(report_df['test_date'])

# Get earliest records
first_test_date_df = report_df.sort_values('test_date').groupby('student_id').first().reset_index()
first_test_date_df.drop(columns=['rn'], inplace=True)
first_test_date_df.rename(columns={'test_date': 'first_test_date'}, inplace=True)

# NOW TO GET THE EXERCISE DATA
# Step 1: Collect Student IDs and Date Ranges
user_df = user_df[user_df['org_name'] == selected_org]

user_df['start_date'] = start_date
user_df['end_date'] = end_date

student_date_ranges = user_df[['student_id', 'start_date', 'end_date']].drop_duplicates()


def create_subquery_for_student_dates(student_date_ranges):
    subquery_parts = []
    for _, row in student_date_ranges.iterrows():
        student_id = int(row['student_id'])
        start_date = row['start_date'].strftime('%Y-%m-%d')
        end_date = row['end_date'].strftime('%Y-%m-%d')
        subquery_part = f"SELECT {student_id} AS student_id, CAST('{start_date}' AS DATE) AS start_date, CAST('{end_date}' AS DATE) AS end_date"
        subquery_parts.append(subquery_part)
    
    # Check if subquery_parts is empty and set a default subquery to avoid syntax errors
    if not subquery_parts:
        # This ensures the subquery is never empty and avoids the INNER JOIN () syntax error
        subquery = "SELECT 1 AS student_id, CAST('1970-01-01' AS DATE) AS start_date, CAST('1970-01-01' AS DATE) AS end_date WHERE 1=0"
    else:
        # If subquery_parts is not empty, join them with UNION ALL as originally intended
        subquery = " UNION ALL ".join(subquery_parts)

    return subquery


# Create subquery
subquery = create_subquery_for_student_dates(student_date_ranges)


def cached_dynamic_query(subquery):
    engine = get_database_engine()  # Ensure the engine is created fresh inside the function
    final_query = f"""
    SELECT r.student_id, r.title, r.start_time, r.smart_duration, r.total_duration, r.intervals, r.goal_minutes 
    FROM oval_athena.report_exercise_results r
    INNER JOIN ({subquery}) s ON r.student_id = s.student_id 
    AND r.start_time BETWEEN s.start_date AND s.end_date
    """
    df = pd.read_sql(final_query, engine)
    return df

# Keep the generation of subquery as is
subquery = create_subquery_for_student_dates(student_date_ranges)

# Now use the cached_dynamic_query function to execute and cache your dynamic query
exercise_df = cached_dynamic_query(subquery)

# Assuming exercise_df is already defined and contains the necessary columns
# Ensure start_time is a datetime column
exercise_df['start_time'] = pd.to_datetime(exercise_df['start_time'])

# Add a week number column based on start_time
exercise_df['week'] = exercise_df['start_time'].dt.isocalendar().week

def calculate_weekly_adherence(sub_df, goal_minutes, days_in_week):
    if sub_df.empty:
        return np.nan
    # Only include "Smart Zone" activities for adherence calculation
    smart_zone_df = sub_df[sub_df['title'] == 'Smart Zone']
    smart_duration_sum = smart_zone_df['smart_duration'].sum() / 60  # Convert seconds to minutes
    
    # Prorate the goal_minutes based on the actual days in the week within the date range
    total_goal_minutes = (goal_minutes / 7) * days_in_week
    adherence = (smart_duration_sum / total_goal_minutes) * 100 if total_goal_minutes > 0 else 0
    basic_adherence = (smart_duration_sum / 90) * 100 if total_goal_minutes > 0 else 0
    
    return min(adherence, 100), min(basic_adherence, 100), smart_duration_sum  # Also return smart_duration_sum


def calculate_adherence_for_range_debug(df, start_date, end_date, total_days, student_id=152):
    week_adherences = []
    detailed_adherence = []

    week_number = 0  # Initialize week number
    current_week_start = start_date
    while current_week_start <= end_date:
        current_week_end = min(current_week_start + timedelta(days=6), end_date)
        current_week_start = pd.to_datetime(current_week_start)
        current_week_end = pd.to_datetime(current_week_end)
        week_df = df[(df['start_time'] >= current_week_start) & (df['start_time'] <= current_week_end)]
        days_in_week = (current_week_end - current_week_start).days + 1

        week_number += 1  # Increment week number for each loop iteration

        # Check if there's activity data for the week; if not, default values to 0
        if not week_df.empty:
            goal_minutes = week_df.iloc[0]['goal_minutes']
            week_adherence, basic_week_adherence, smart_duration_sum = calculate_weekly_adherence(week_df, goal_minutes, days_in_week)
        else:
            goal_minutes = 0  # Default to 0 or use a predefined goal if applicable
            week_adherence, basic_week_adherence, smart_duration_sum = 0, 0, 0

        week_adherences.append((week_adherence, basic_week_adherence, days_in_week))

        # Append detailed adherence info including smart_duration_sum
        detailed_adherence.append({
            'week_number': week_number,
            'week_start_date': current_week_start,
            'week_end_date': current_week_end,
            'days_in_week': days_in_week,
            'goal_minutes': goal_minutes,
            'week_adherence': week_adherence,
            'basic_week_adherence': basic_week_adherence,
            'smart_duration_sum': smart_duration_sum
        })

        current_week_start += timedelta(days=7)  # Proceed to the next week

    detailed_adherence_df = pd.DataFrame(detailed_adherence)
    # Calculate the weighted average of weekly adherences, including weeks with 0 adherence
    weighted_adherence = np.average([ad['week_adherence'] for ad in detailed_adherence], weights=[ad['days_in_week'] for ad in detailed_adherence]) if detailed_adherence else np.nan
    weighted_basic_adherence = np.average([ad['basic_week_adherence'] for ad in detailed_adherence], weights=[ad['days_in_week'] for ad in detailed_adherence]) if detailed_adherence else np.nan

    if not detailed_adherence_df.empty and df.iloc[0]['student_id'] == 152:  # Make sure this condition correctly identifies Sam Speir
        print(f"Detailed weekly adherence for {student_id}:")
        print(detailed_adherence_df)
        return weighted_adherence, weighted_basic_adherence, detailed_adherence_df

    return weighted_adherence, weighted_basic_adherence, detailed_adherence_df



adherence_results = []
detailed_adherence_df_results = pd.DataFrame()  # Placeholder for detailed adherence info
detailed_adherence_for_sam = None  # Placeholder for Sam's detailed adherence info

for student_id, student_df in exercise_df.groupby('student_id'):
    # Fetch start and end dates from merged_common based on student_id
    # Here, .iloc[0] is used to ensure we're getting the first matching row in case there are multiple (which shouldn't normally happen if your data is consistent)
    start_date = start_date
    end_date = end_date
    
    # Assuming calculate_adherence_for_range_debug is a function you have defined elsewhere that calculates adherence
    # And assuming total_days is defined somewhere in your code based on the start and end dates
    total_days = (end_date - start_date).days + 1  # Adjust according to your specific requirements

    if student_id == 152:
        # Calculate adherence for Sam with debug information
        student_adherence, student_basic_adherence, detailed_adherence_for_sam = calculate_adherence_for_range_debug(student_df, start_date, end_date, total_days, student_id=152)
    else:
        # Calculate adherence for other students
        student_adherence, student_basic_adherence, detailed_adherence_df = calculate_adherence_for_range_debug(student_df, start_date, end_date, total_days)
        detailed_adherence_df['student_id'] = student_id  # Add student_id to the detailed adherence info

    adherence_results.append({'student_id': student_id, 'adherence%': student_adherence, 'basic_adherence%': student_basic_adherence})
    detailed_adherence_df_results = pd.concat([detailed_adherence_df_results, detailed_adherence_df])

adherence_df = pd.DataFrame(adherence_results)

aggregations = {
    'smart_duration': lambda x: np.round(x[exercise_df.loc[x.index, 'title'] == 'Smart Zone'].fillna(0).sum() / 60).astype(int),
    'intervals': lambda x: (x >= 1).sum(),  # Correctly adjusted line
    'total_duration': lambda x: np.round(x[exercise_df.loc[x.index, 'title'] == 'Smart Zone'].fillna(0).sum() / 60).astype(int)
}

raw_adherence = exercise_df.groupby('student_id').agg(aggregations).reset_index()

adherence_df = adherence_df.merge(raw_adherence, on='student_id', how='left')

# save to a csv
adherence_df.to_csv('adherence_report.csv', index=False)
detailed_adherence_df_results.to_csv('detailed_adherence_report.csv', index=False)

# remove any weeks for student_ids that are prior to their start date in the first_test_date_df
detailed_adherence_df_results = detailed_adherence_df_results.merge(first_test_date_df, on='student_id', how='left')
detailed_adherence_df_results = detailed_adherence_df_results[detailed_adherence_df_results['week_start_date'] >= detailed_adherence_df_results['first_test_date']]

# Plotting the adherence data
# Start with calculated the percentage of users who completed their goal for each week
group_weekly_adherence_completed_percentage = (
    detailed_adherence_df_results.groupby('week_start_date')
    .apply(lambda x: (x['week_adherence'] == 100).mean() * 100)
    .reset_index(name='completed_percentage')
)

# Now do it for basic adherence
group_weekly_basic_adherence_completed_percentage = (
    detailed_adherence_df_results.groupby('week_start_date')
    .apply(lambda x: (x['basic_week_adherence'] == 100).mean() * 100)
    .reset_index(name='completed_basic_percentage')
)

# Now we need to get the percentage of people who did any smart time at all during each week
group_weekly_any_smart_time_percentage = (
    detailed_adherence_df_results.groupby('week_start_date')
    .apply(lambda x: (x['smart_duration_sum'] > 0).mean() * 100)
    .reset_index(name='any_smart_time_percentage')
)

# Merge the three dataframes on 'week_start_date'
merged_df = (
    group_weekly_adherence_completed_percentage
    .merge(group_weekly_basic_adherence_completed_percentage, on='week_start_date')
    .merge(group_weekly_any_smart_time_percentage, on='week_start_date')
)

# Melt the dataframe to long format for Seaborn
melted_df = merged_df.melt(
    id_vars='week_start_date',
    value_vars=['any_smart_time_percentage', 'completed_basic_percentage', 'completed_percentage'],
    var_name='Metric',
    value_name='Percentage'
)

# Map metric names to more readable labels
metric_labels = {
    'any_smart_time_percentage': 'Any Smart Time',
    'completed_percentage': 'Completed Personal Adherence Goal',
    'completed_basic_percentage': 'Completed 90min Adherence Goal'
}
melted_df['Metric'] = melted_df['Metric'].map(metric_labels)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")  # Options: darkgrid, whitegrid, dark, white, ticks
sns.set_context("talk")  # Options: paper, notebook, talk, poster

# Create the plot
plt.figure(figsize=(12, 6))

sns.lineplot(
    data=melted_df,
    x='week_start_date',
    y='Percentage',
    hue='Metric',
    style='Metric',
    markers=True,
    dashes=False,
    linewidth=2.5
)

# Customize the plot
plt.title(f'{selected_org} - Weekly Adherence Metrics Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Week Start Date', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.xticks(rotation=45)

# rotate the x-axis labels for better readability
plt.xticks(rotation=45) 

# Format the x-axis date labels
date_form = DateFormatter("%Y-%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)

plt.legend(title='Metric', fontsize=10, title_fontsize=12, loc='upper left')

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Add gridlines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save and show the plot
plt.savefig(f'{selected_org} weekly_adherence_metrics.png', dpi=300, bbox_inches='tight')
plt.show()