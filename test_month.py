import pandas as pd

# Create sample dates throughout a year
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
df = pd.DataFrame({'date': dates})
df['month'] = df['date'].dt.month

# Show unique month values
print("Month feature returns integer values from 1-12:")
print("Unique month values:", sorted(df['month'].unique()))
print()

# Show some examples
print("Examples of date -> month mapping:")
sample_dates = ['2023-01-15', '2023-03-10', '2023-06-20', '2023-09-05', '2023-12-25']
for date_str in sample_dates:
    date_obj = pd.to_datetime(date_str)
    month_val = date_obj.month
    print(f"{date_str} -> month = {month_val}")

print()
print("The 'month' feature extracts the month number (1-12) from each date:")
print("January = 1, February = 2, ..., December = 12")
