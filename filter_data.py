

def drop_rows(df, column_, start_, end_):

    exclude_ = range(start_, end_ + 1)


    filtered_df = df[~df[column_].isin(exclude_)]
    
    return filtered_df


def check_filtering(df_filtered, df_original, row_to_check):
  
    if df_filtered.shape[0] < df_original.shape[0]:
        print("Rows have been filtered.")
    else:
        print("No rows have been filtered.")

    # Check if a specific year has been removed
    if row_to_check is not None:
        try:
            # Attempt to access a row with the specified year
            row = df_filtered[df_filtered['Year'] == row_to_check].iloc[0]
            print(f"The Row {row_to_check} is still present in the DataFrame.")
        except IndexError:
            print(f"The Row {row_to_check} has been filtered out of the DataFrame.")

# Example usage: