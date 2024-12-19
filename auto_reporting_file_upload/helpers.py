from datetime import datetime, date, timedelta
from sqlalchemy import create_engine
import pandas as pd
import os
import numpy as np


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

def get_unique_org_names():
    engine = get_database_engine()
    query = "SELECT DISTINCT org_name FROM oval_athena.users"
    df = pd.read_sql(query, engine)
    return df['org_name'].tolist()


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

def calculate_weekly_adherence(sub_df, goal_minutes):
    if sub_df.empty:
        return np.nan
    # Only include "Smart Zone" activities for adherence calculation
    smart_zone_df = sub_df[sub_df['title'] == 'Smart Zone']
    smart_duration_sum = smart_zone_df['smart_duration'].sum() / 60  # Convert seconds to minutes
    
    # Prorate the goal_minutes based on the actual days in the week within the date range
    adherence = (smart_duration_sum / goal_minutes) * 100 if goal_minutes > 0 else 0
    basic_adherence = (smart_duration_sum / 90) * 100 if goal_minutes > 0 else 0
    interval_adherence = (sub_df['intervals'] >= 1).sum() * 100
    return min(adherence, 100), min(basic_adherence, 100), smart_duration_sum, min(100, interval_adherence)  # Also return smart_duration_sum


def calculate_adherence_for_range_debug(df, start_date, end_date, total_days, student_id=152):
    week_adherences = []
    detailed_adherence = []

    week_number = 0  # Initialize week number
    current_week_start = start_date
    while current_week_start <= (end_date - timedelta(days=1)):
        current_week_end = current_week_start + timedelta(days=6)
        current_week_start = pd.to_datetime(current_week_start)
        current_week_end = pd.to_datetime(current_week_end)
        week_df = df[(df['start_time'] >= current_week_start) & (df['start_time'] <= current_week_end)]
        days_in_week = (current_week_end - current_week_start).days + 1

        week_number += 1  # Increment week number for each loop iteration

        # Check if there's activity data for the week; if not, default values to 0
        if not week_df.empty:
            goal_minutes = week_df.iloc[0]['goal_minutes']
            week_adherence, basic_week_adherence, smart_duration_sum, interval_adherence = calculate_weekly_adherence(week_df, goal_minutes)
        else:
            goal_minutes = 0  # Default to 0 or use a predefined goal if applicable
            week_adherence, basic_week_adherence, smart_duration_sum, interval_adherence = 0, 0, 0, 0

        week_adherences.append((week_adherence, basic_week_adherence, interval_adherence, days_in_week))

        # Append detailed adherence info including smart_duration_sum
        detailed_adherence.append({
            'week_number': week_number,
            'week_start_date': current_week_start,
            'week_end_date': current_week_end,
            'days_in_week': days_in_week,
            'goal_minutes': goal_minutes,
            'week_adherence': week_adherence,
            'basic_week_adherence': basic_week_adherence,
            'smart_duration_sum': smart_duration_sum,
            'interval_adherence': interval_adherence
        })

        current_week_start += timedelta(days=7)  # Proceed to the next week

    detailed_adherence_df = pd.DataFrame(detailed_adherence)
    # Calculate the weighted average of weekly adherences, including weeks with 0 adherence
    weighted_adherence = np.average([ad['week_adherence'] for ad in detailed_adherence], weights=[ad['days_in_week'] for ad in detailed_adherence]) if detailed_adherence else np.nan
    weighted_basic_adherence = np.average([ad['basic_week_adherence'] for ad in detailed_adherence], weights=[ad['days_in_week'] for ad in detailed_adherence]) if detailed_adherence else np.nan
    weighted_interval_adherence = np.average([ad['interval_adherence'] for ad in detailed_adherence], weights=[ad['days_in_week'] for ad in detailed_adherence]) if detailed_adherence else np.nan

    if not detailed_adherence_df.empty and df.iloc[0]['student_id'] == 152:  # Make sure this condition correctly identifies Sam Speir
        print(f"Detailed weekly adherence for {student_id}:")
        print(detailed_adherence_df)
        return weighted_adherence, weighted_basic_adherence, detailed_adherence_df

    return weighted_adherence, weighted_basic_adherence, detailed_adherence_df, weighted_interval_adherence