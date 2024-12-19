import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from datetime import datetime, date, timedelta
from matplotlib.dates import DateFormatter
from helpers import get_database_engine, cached_query, get_unique_org_names, create_subquery_for_student_dates, cached_dynamic_query, calculate_weekly_adherence, calculate_adherence_for_range_debug

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


# Use the function to fetch unique organization names
org_names = get_unique_org_names()

org_names = [org_name if org_name is not None else 'Unknown' for org_name in org_names]

org_names = [str(org_name) for org_name in org_names if org_name is not None]

bad_orgs = ["Unknown", 'Deleted users', 'DUMMY Org', 'Elexr Team', 
         'Test', 'To be deleted']

all_orgs = [org for org in org_names if org not in bad_orgs]


# MAKE YOUR SELECTIONS HERE
selected_org = all_orgs
start_date = datetime.strptime('2024-01-01', '%Y-%m-%d')
end_date = datetime.now()

today = pd.Timestamp.now().normalize()  # Returns today's date as a normalized Timestamp (midnight)
current_week_end_date = today - pd.Timedelta(days=(today.weekday() + 1))
current_week_start_date = current_week_end_date - pd.Timedelta(days=6)

last_week_end_date = current_week_start_date - pd.Timedelta(days=1)
last_week_start_date = last_week_end_date - pd.Timedelta(days=6)

total_days = (end_date - start_date).days
remainder = total_days % 7

if remainder != 0:
    # Adjust end_date so that the difference is divisible by 7.
    # This will round DOWN to the nearest multiple of 7
    total_days = total_days - remainder
    end_date = start_date + timedelta(days=total_days)

# Assuming 'test_date' is the column to determine earliest and latest records
# Convert 'test_date' to datetime if not already
report_df['test_date'] = pd.to_datetime(report_df['test_date'])

# Get earliest records
first_test_date_df = report_df.sort_values('test_date').groupby('student_id').first().reset_index()
first_test_date_df.drop(columns=['rn'], inplace=True)
first_test_date_df.rename(columns={'test_date': 'first_test_date'}, inplace=True)

print(type(selected_org))

# NOW TO GET THE EXERCISE DATA
# Step 1: Collect Student IDs and Date Ranges
if isinstance(selected_org, str):
    user_df = user_df[user_df['org_name'] == selected_org]
else:
    user_df = user_df[user_df['org_name'].isin(selected_org)]


user_df['start_date'] = start_date
user_df['end_date'] = end_date

student_date_ranges = user_df[['student_id', 'start_date', 'end_date']].drop_duplicates()



# Create subquery
subquery = create_subquery_for_student_dates(student_date_ranges)

# Keep the generation of subquery as is
subquery = create_subquery_for_student_dates(student_date_ranges)

# Now use the cached_dynamic_query function to execute and cache your dynamic query
exercise_df = cached_dynamic_query(subquery)

# Assuming exercise_df is already defined and contains the necessary columns
# Ensure start_time is a datetime column
exercise_df['start_time'] = pd.to_datetime(exercise_df['start_time'])

# Add a week number column based on start_time
exercise_df['week'] = exercise_df['start_time'].dt.isocalendar().week



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
        student_adherence, student_basic_adherence, detailed_adherence_df, weighted_interval_adherence = calculate_adherence_for_range_debug(student_df, start_date, end_date, total_days)
        detailed_adherence_df['student_id'] = student_id  # Add student_id to the detailed adherence info

    adherence_results.append({'student_id': student_id, 'adherence%': student_adherence, 'basic_adherence%': student_basic_adherence, 'interval_adherence%': weighted_interval_adherence})
    detailed_adherence_df_results = pd.concat([detailed_adherence_df_results, detailed_adherence_df])

adherence_df = pd.DataFrame(adherence_results)



current_week_adherence_results = []
current_week_detailed_adherence_df_results = pd.DataFrame()  # Placeholder for detailed adherence info

for student_id, student_df in exercise_df.groupby('student_id'):
    # Fetch start and end dates from merged_common based on student_id
    
    # Assuming calculate_adherence_for_range_debug is a function you have defined elsewhere that calculates adherence
    # And assuming total_days is defined somewhere in your code based on the start and end dates
    total_days = (current_week_end_date - current_week_start_date).days + 1  # Adjust according to your specific requirements

    # Calculate adherence for other students
    student_adherence, student_basic_adherence, detailed_adherence_df, weighted_interval_adherence = calculate_adherence_for_range_debug(student_df, current_week_start_date, current_week_end_date, total_days)
    detailed_adherence_df['student_id'] = student_id  # Add student_id to the detailed adherence info

    current_week_adherence_results.append({'student_id': student_id, 'adherence%': student_adherence, 'basic_adherence%': student_basic_adherence, 'interval_adherence%': weighted_interval_adherence})
    current_week_detailed_adherence_df_results = pd.concat([current_week_detailed_adherence_df_results, detailed_adherence_df])

current_week_adherence_df = pd.DataFrame(current_week_adherence_results)



last_week_adherence_results = []
last_week_detailed_adherence_df_results = pd.DataFrame()  # Placeholder for detailed adherence info

for student_id, student_df in exercise_df.groupby('student_id'):
    # Fetch start and end dates from merged_common based on student_id
    
    # Assuming calculate_adherence_for_range_debug is a function you have defined elsewhere that calculates adherence
    # And assuming total_days is defined somewhere in your code based on the start and end dates
    total_days = (last_week_end_date - last_week_start_date).days + 1  # Adjust according to your specific requirements

    # Calculate adherence for other students
    student_adherence, student_basic_adherence, detailed_adherence_df, weighted_interval_adherence = calculate_adherence_for_range_debug(student_df, last_week_start_date, last_week_end_date, total_days)
    detailed_adherence_df['student_id'] = student_id  # Add student_id to the detailed adherence info

    last_week_adherence_results.append({'student_id': student_id, 'adherence%': student_adherence, 'basic_adherence%': student_basic_adherence, 'interval_adherence%': weighted_interval_adherence})
    last_week_detailed_adherence_df_results = pd.concat([last_week_detailed_adherence_df_results, detailed_adherence_df])

last_week_adherence_df = pd.DataFrame(last_week_adherence_results)



# apply the aggregation functions to the dataframes
aggregations = {
    'smart_duration': lambda x: np.round(x[exercise_df.loc[x.index, 'title'] == 'Smart Zone'].fillna(0).sum() / 60).astype(int),
    'intervals': lambda x: (x >= 1).sum(),  # Correctly adjusted line
    'total_duration': lambda x: np.round(x[exercise_df.loc[x.index, 'title'] == 'Smart Zone'].fillna(0).sum() / 60).astype(int)
}

raw_adherence = exercise_df.groupby('student_id').agg(aggregations).reset_index()
adherence_df = adherence_df.merge(raw_adherence, on='student_id', how='left')

# do it for the current week
current_week_raw_adherence = exercise_df[(exercise_df['start_date'] >= current_week_start_date) & (exercise_df['start_date'] <= current_week_end_date)].groupby('student_id').agg(aggregations).reset_index()
current_week_adherence_df = current_week_adherence_df.merge(raw_adherence, on='student_id', how='left')
current_week_adherence_df[["smart_duration","intervals","total_duration"]] = (
    current_week_adherence_df[["smart_duration","intervals","total_duration"]].fillna(0, inplace=True)
)

# do it for the last week
last_week_raw_adherence = exercise_df[(exercise_df['start_date'] >= last_week_start_date) & (exercise_df['start_date'] <= last_week_end_date)].groupby('student_id').agg(aggregations).reset_index()
last_week_adherence_df = last_week_adherence_df.merge(raw_adherence, on='student_id', how='left')
last_week_adherence_df[["smart_duration","intervals","total_duration"]] = (
    last_week_adherence_df[["smart_duration","intervals","total_duration"]].fillna(0, inplace=True)
)


# save to a csv
adherence_df.to_csv('adherence_report.csv', index=False)
detailed_adherence_df_results.to_csv('detailed_adherence_report.csv', index=False)
current_week_adherence_df.to_csv('current_week_adherence_report.csv', index=False)
current_week_detailed_adherence_df_results.to_csv('current_week_detailed_adherence_report.csv', index=False)
last_week_adherence_df.to_csv('last_week_adherence_report.csv', index=False)
last_week_detailed_adherence_df_results.to_csv('last_week_detailed_adherence_report.csv', index=False)


# remove any weeks for student_ids that are prior to their start date in the first_test_date_df
detailed_adherence_df_results = detailed_adherence_df_results.merge(first_test_date_df, on='student_id', how='left')
detailed_adherence_df_results = detailed_adherence_df_results[detailed_adherence_df_results['week_start_date'] >= detailed_adherence_df_results['first_test_date']]


# Plotting the adherence data
# Time to plot the current week bar chart first
# Set a theme for aesthetics
sns.set_theme(style="whitegrid")

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create the bar plot
sns.barplot(
    x="name", 
    y="adherence%", 
    data=current_week_adherence_df, 
    palette="viridis", 
    ax=ax
)

# Customize labels and title
ax.set_xlabel("Name", fontsize=14, labelpad=10)
ax.set_ylabel("Adherence Percentage (%)", fontsize=14, labelpad=10)
ax.set_title(f'{selected_org} Current Week Adherence', fontsize=16, fontweight='bold', pad=15)

# Rotate x-axis labels for better readability if there are many names
plt.xticks(rotation=90, ha='center', fontsize=6)

# save and show plot
sns.despine()
plt.tight_layout()
plt.savefig(f'{selected_org} current_week_adherence.png', dpi=300, bbox_inches='tight')
plt.show()


# plot the double bar chart

# Add a column to distinguish the weeks
current_week = current_week_adherence_df.assign(Week='Current Week')
last_week = last_week_adherence_df.assign(Week='Last Week')

# Combine the two DataFrames into one
combined_df = pd.concat([current_week, last_week], ignore_index=True)

# Set a theme for aesthetics
sns.set_theme(style="whitegrid")

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Extract categories (make sure both DataFrames share the same categories)
categories = combined_df["name"].unique()

# Create the grouped bar plot using 'hue' to differentiate current vs. last week
sns.barplot(
    x="name", 
    y="adherence%", 
    hue="Week",   # differentiate bars by week
    data=combined_df,
    order=categories,
    palette=["#2C7BB6", "#D7191C"],  # Two contrasting colors, or use a named palette
    ax=ax
)

# Set the ticks at integer positions matching each category
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=90, ha='center', fontsize=6)

# Customize labels and title
ax.set_xlabel("Name", fontsize=14, labelpad=10)
ax.set_ylabel("Adherence Percentage (%)", fontsize=14, labelpad=10)
ax.set_title(f'{selected_org} Current vs Last Week Adherence', fontsize=16, fontweight='bold', pad=15)

# save and display the plot
sns.despine()
plt.tight_layout()
plt.savefig(f'{selected_org}_current_vs_last_week_adherence.png', dpi=300, bbox_inches='tight')
plt.show()


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

# Now do it for interval adherence
group_weekly_interval_adherence_completed_percentage = (
    detailed_adherence_df_results.groupby('week_start_date')
    .apply(lambda x: (x['interval_adherence'] == 100).mean() * 100)
    .reset_index(name='completed_interval_percentage')
)

# Now we need to count the number of unique users who are being tracked each week
group_weekly_unique_users = (
    detailed_adherence_df_results.groupby('week_start_date')
    .apply(lambda x: x['student_id'].nunique())
    .reset_index(name='user_count')
)


# Merge the three dataframes on 'week_start_date'
merged_df = (
    group_weekly_adherence_completed_percentage
    .merge(group_weekly_basic_adherence_completed_percentage, on='week_start_date')
    .merge(group_weekly_any_smart_time_percentage, on='week_start_date')
    .merge(group_weekly_interval_adherence_completed_percentage, on='week_start_date')
    .merge(group_weekly_unique_users, on='week_start_date')
)

# Melt the dataframe to long format for Seaborn
melted_df = merged_df.melt(
    id_vars='week_start_date',
    value_vars=['any_smart_time_percentage', 'completed_basic_percentage', 'completed_percentage', 'completed_interval_percentage'],
    var_name='Metric',
    value_name='Percentage'
)

# Map metric names to more readable labels
metric_labels = {
    'any_smart_time_percentage': 'Any Smart Time',
    'completed_percentage': 'Completed Custom Goal',
    'completed_basic_percentage': 'Completed 90min Goal',
    'completed_interval_percentage': 'Completed Interval Goal'
}
melted_df['Metric'] = melted_df['Metric'].map(metric_labels)


# Set the aesthetic style of the plots
sns.set_style("whitegrid")  # Options: darkgrid, whitegrid, dark, white, ticks
sns.set_context("talk")     # Options: paper, notebook, talk, poster

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Create a twin y-axis for the user count histogram
ax2 = ax1.twinx()

# Plot the line plots on ax1
sns.lineplot(
    data=melted_df,
    x='week_start_date',
    y='Percentage',
    hue='Metric',
    style='Metric',
    markers=True,
    dashes=False,
    linewidth=2.5,
    ax=ax2
)

# Plot the histogram of user_count on ax2
bars = ax1.bar(
    merged_df['week_start_date'],
    merged_df['user_count'],
    color='lightgray',
    alpha=0.3,
    width=6,  # Adjust the width to match the week duration
    label='User Count'
)

# Remove gridlines from ax2
ax1.grid(False)

# Set labels
ax1.set_xlabel('Week Start Date', fontsize=14)
ax2.set_ylabel('Percentage (%)', fontsize=14)
ax1.set_ylabel('User Count', fontsize=14)

# Set the title
if selected_org == all_orgs:
    ax1.set_title('All Organizations - Weekly Adherence Metrics Over Time', fontsize=16, fontweight='bold')
else:
    ax1.set_title(f'{selected_org} - Weekly Adherence Metrics Over Time', fontsize=16, fontweight='bold')

# Format the x-axis date labels to show only the first week of each month
locator = mdates.MonthLocator()
formatter = mdates.DateFormatter('%Y-%m-%d')
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)
ax2.tick_params(axis='x', rotation=45)

# Adjust gridlines to be less visible on ax1
ax2.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.3)

# Adjust the legend
# Get the handles and labels from ax1
handles1, labels1 = ax2.get_legend_handles_labels()

# Create a custom legend handle for the bar chart
user_count_handle = Patch(facecolor='lightgray', edgecolor='lightgray', alpha=0.3, label='User Count')

# Combine the handles and labels
handles = handles1 + [user_count_handle]
labels = labels1 + ['User Count']

# Place the legend outside the plot
ax2.legend(handles, labels, title='Metric', fontsize=9, title_fontsize=10,
           loc='upper left', bbox_to_anchor=(1.10, 1), borderaxespad=0., frameon=False)

# Adjust layout to prevent clipping of tick-labels and legend
plt.tight_layout()

# Save and show the plot
if selected_org == all_orgs:
    plt.savefig('all orgs weekly_adherence_metrics.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(f'{selected_org} weekly_adherence_metrics.png', dpi=300, bbox_inches='tight')

plt.show()

#print(adherence_df['adherence%'].mean())