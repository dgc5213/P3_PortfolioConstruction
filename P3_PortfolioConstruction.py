import matplotlib.dates as mdates  # Import the dates module
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def readCSV(path_csv):

    df = pd.read_csv(path_csv)
    return df


def readExcel(path_xlsx):
    df = pd.read_excel(path_xlsx)
    return df


def check_missing_values(df, plot=False):
    # Check for missing values
    missing_data = df.isna().sum()

    # Print missing value counts
    print("Missing Value Counts:")
    print(missing_data)

    if plot:
        # Plot missing value counts
        plt.figure(figsize=(10, 6))
        missing_data.plot(kind='bar')
        plt.title("Missing Value Counts")
        plt.xlabel("Columns")
        plt.ylabel("Missing Value Count")
        plt.xticks(rotation=45)
        # plt.show()
        plt.pause(1)
        plt.close()




def plot_portfolio_weights(sorted_data, title, filename):
    # Get the top 5 and bottom 5 weights
    top_5 = sorted_data.nlargest(5, 'Weights')
    bottom_5 = sorted_data.nsmallest(5, 'Weights')

    # Create a plot
    plt.figure(figsize=(12, 6))

    # Plot all industries
    plt.bar(sorted_data['Industry'], sorted_data['Weights'], color='b', label='All Industries')

    # Highlight the top 5 weights in green
    plt.bar(top_5['Industry'], top_5['Weights'], color='g', label='Top 5 Weights')

    # Highlight the bottom 5 weights in red
    plt.bar(bottom_5['Industry'], bottom_5['Weights'], color='r', label='Bottom 5 Weights')

    plt.xlabel('Industry')
    plt.ylabel('Weights')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()
    plt.pause(1)
    plt.close()



def annualized_volatility(returns):
    return np.std(returns) * np.sqrt(12)

if __name__ == '__main__':
    print ("---------------------start ---------------\n")



###----------!!!!----Please edit the raw data path and choose correct file format----------------------
    raw_csv = "1_Rawdata/49_Industry_Portfolios_AverageValueWeightedReturn_Monthly.csv"
    df_raw = readCSV(raw_csv)
    print(df_raw)
    df=df_raw



    ###----------------------- Data Preparation ----------------
    ### Convert the 'yearmonth' column to datetime format
    df['yearmonth'] = pd.to_datetime(df['yearmonth'], format='%Y%m')
    ### Set the 'yearmonth' column as the index
    df.set_index('yearmonth', inplace=True)

    check_missing_values(df, plot=True)    ## The result shows: No missing value

    ### the first 10 years (January 1995 – December 2004) as a “seed period”
    start_date = '1995-01-01'
    end_date = '2004-12-31'
    seed_data = df[(df.index >= start_date) & (df.index <= end_date)]
    seed_data.to_csv('2_WIP/seed_data.csv')

    covariance_matrix = seed_data.cov()
    print(covariance_matrix)
    covariance_matrix.to_csv('2_WIP/covariance_matrix.csv')


    ##----------set up an optimization to calculate the minimum-variance portfolio as of the close of 12/31/2004--------
    ### =========== minimum-variance portfolio=========

    # Define constraints for the optimization
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Fully invested
    ##1: Minimum-variance portfolio optimization
    initial_weights = np.ones(len(seed_data.columns)) / len(seed_data.columns)  # Initial equal weights
    objective = lambda weights: np.dot(weights.T, np.dot(covariance_matrix, weights))
    min_var_portfolio = minimize(objective, initial_weights, method='SLSQP', constraints=constraints)


    ##2: Minimum-variance portfolio with long-only constraints
    n_assets = len(seed_data.columns)  # Number of assets
    initial_weights = np.ones(n_assets) / n_assets # Initial equal weights
    objective = lambda weights: np.dot(weights.T, np.dot(covariance_matrix, weights))
    constraints_long_only = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}, # Fully invested
        {'type': 'ineq', 'fun': lambda weights: weights}  # Long-only constraint
    )
    # Bounds for asset weights (0 to 1 for long-only)
    bounds = tuple((0, 1) for asset in range(n_assets))
    min_var_portfolio_long_only = minimize(objective, initial_weights,method='SLSQP', bounds=bounds,
                                           constraints=constraints_long_only)

    ##3: Inverse-volatility portfolio
    # Calculate the inverse of historical volatility (1/sigma) for each asset
    inverse_volatility = 1 / np.sqrt(np.diag(covariance_matrix))
    # Normalize inverse-volatility to sum to 1 (proportional weights)
    inverse_volatility = inverse_volatility / np.sum(inverse_volatility)


    # Print the results
    # Calculate the total sum of weights for each Portfolio
    total_weight_min_var_portfolio = round(min_var_portfolio.x.sum(),1)
    total_weight_min_var_portfolio_long_only = round(min_var_portfolio_long_only.x.sum(),1)
    total_weight_inverse_volatility = round(np.sum(inverse_volatility),1)

    industry_names = seed_data.columns
    print("Minimum-Variance Portfolio (fully invested but may include short positions):")
    print("Industry:", industry_names)
    print("Weights:", min_var_portfolio.x)
    print("Total Weight:", total_weight_min_var_portfolio)
    print("Portfolio Variance:", min_var_portfolio.fun)

    print("\nMinimum-Variance Portfolio (long-only):")
    print("Industry:", industry_names)
    print("Weights:", min_var_portfolio_long_only.x)
    print("Total Weight:", total_weight_min_var_portfolio_long_only)
    print("Portfolio Variance:", min_var_portfolio_long_only.fun)

    print("\nInverse-Volatility Portfolio:")
    print("Industry:", industry_names)
    print("Weights (Proportional to Inverse Volatility):")
    print(inverse_volatility)
    print("Total Weight (Inverse-Volatility Portfolio):", total_weight_inverse_volatility)

    # Create dataframes for each portfolio
    results_min_var_portfolio = pd.DataFrame({'Industry': industry_names, 'Weights': min_var_portfolio.x})
    results_min_var_portfolio_long_only = pd.DataFrame(
        {'Industry': industry_names, 'Weights': min_var_portfolio_long_only.x})

    results_inverse_volatility_portfolio = pd.DataFrame({'Industry': industry_names, 'Weights': inverse_volatility})

    # Save the dataframes to Excel files
    results_min_var_portfolio.to_excel('2_WIP/results_min_var_portfolio_fully_invested.xlsx', index=False)
    results_min_var_portfolio_long_only.to_excel('2_WIP/results_min_var_portfolio_long_only.xlsx', index=False)
    results_inverse_volatility_portfolio.to_excel('2_WIP/results_inverse_volatility_portfolio.xlsx', index=False)




    ###===================
    ###For each strategy, highlight the top 5 weights in green and the bottom 5 weights in red to allow easy comparison.

    # Sort the data by weights
    sorted_data_min_var = results_min_var_portfolio.sort_values('Weights')
    sorted_data_min_var_long_only = results_min_var_portfolio_long_only.sort_values('Weights')
    sorted_data_inverse_volatility = results_inverse_volatility_portfolio.sort_values('Weights')

    # Plot for Minimum-Variance Portfolio (fully invested but may include short positions)
    plot_portfolio_weights(sorted_data_min_var, 'Minimum-Variance Portfolio (Fully invested)',
                           '2_IMG/min_var_portfolio_weights_plot.png')

    # Plot for Minimum-Variance Portfolio (Long-Only)
    plot_portfolio_weights(sorted_data_min_var_long_only, 'Minimum-Variance Portfolio (Long-Only)',
                           '2_IMG/min_var_portfolio_long_only_weights_plot.png')

    # Plot for Inverse-Volatility Portfolio
    plot_portfolio_weights(sorted_data_inverse_volatility, 'Inverse-Volatility Portfolio',
                           '2_IMG/inverse_volatility_weights_plot.png')




    ###===================================================================================
    ###======= 3B) in-sample effectiveness================================================
    ## For each of the 3 low-volatility portfolios created earlier,
    ## show the in-sample annualized return volatility for the period January 1995 through December 2004.
    ### ==================================================================================================

    # Calculate portfolio returns for the three low-volatility portfolios
    portfolio_weights = [min_var_portfolio.x, min_var_portfolio_long_only.x,inverse_volatility]
    portfolio_returns = np.dot(seed_data, np.array(portfolio_weights).T)

    # Calculate annualized volatility for each portfolio
    portfolio_volatility = [annualized_volatility(portfolio_returns[:, i]) for i in range(len(portfolio_weights))]
    # Print or store the results
    print("\n\nIn-Sample Annualized Volatility (Jan 1995 - Dec 2004):")
    for i, portfolio_name in enumerate(
            ["Minimum Variance Portfolio (Fully Invested)", "Minimum Variance Portfolio (Long-Only)", "Inverse-Volatility Portfolio"]):
        print(f"{portfolio_name}: {portfolio_volatility[i]:.4f}")

    ###================
    # Create a Pandas DataFrame for in_sample_portfolio_returns  AND save to csv for 3E
    in_sample_portfolio_return_df = pd.DataFrame({'Date': seed_data.index,
                                                      "Minimum Variance Portfolio (Fully Invested)": portfolio_returns[:, 0],
                                                      "Minimum Variance Portfolio (Long-Only)": portfolio_returns[:, 1],
                                                      "Inverse-Volatility Portfolio": portfolio_returns[:, 2]})
    in_sample_portfolio_return_df.to_csv('2_WIP/in_sample_portfolio_returns_df.csv', index=False)




    ###====================================================================================================
    ###======= 3C) out-of-sample effectiveness
    ##  For each of the 3 low-volatility portfolios created earlier,
    ### show the out-of-sample annualized return volatility for the period January 2005 through July 2023.
    ### ==================================================================================================


    # Define the out-of-sample period
    out_of_sample_start_date = '2005-01-01'
    out_of_sample_end_date = '2023-07-31'

    # Filter the data for the out-of-sample period
    out_of_sample_data = df[(df.index >= out_of_sample_start_date) & (df.index <= out_of_sample_end_date)]
    out_of_sample_data.to_csv('2_WIP/out_of_sample_data.csv')

    # Calculate portfolio returns for the out-of-sample period
    out_of_sample_portfolio_returns = np.dot(out_of_sample_data, np.array(portfolio_weights).T)






    ####_____________________________
    # Calculate annualized volatility for each portfolio in the out-of-sample period
    out_of_sample_portfolio_volatility = [annualized_volatility(out_of_sample_portfolio_returns[:, i]) for i in
                                          range(len(portfolio_weights))]

    # Print or store the out-of-sample results
    print ("\n\nOut-of-Sample Annualized Volatility (Jan 2005 - Jul 2023):")
    for i, portfolio_name in enumerate(
            ["Minimum Variance Portfolio (Fully Invested)", "Minimum Variance Portfolio (Long-Only)",
             "Inverse-Volatility Portfolio"]):
        print(
            f"{portfolio_name}: {out_of_sample_portfolio_volatility[i]:.4f}")


        ###================
        # Create a Pandas DataFrame for out_of_sample_portfolio_returns  AND save to csv for 3E
    out_of_sample_portfolio_return_df = pd.DataFrame({'Date': out_of_sample_data.index,
                                  "Minimum Variance Portfolio (Fully Invested)": out_of_sample_portfolio_returns[:, 0],
                                  "Minimum Variance Portfolio (Long-Only)": out_of_sample_portfolio_returns[:, 1],
                                  "Inverse-Volatility Portfolio": out_of_sample_portfolio_returns[:, 2]})
    out_of_sample_portfolio_return_df.to_csv('2_WIP/out_of_sample_portfolio_returns_df.csv', index=False)






    ###====================================================================================================
    ####======= 3D) the value of re-optimization
    ### Now create an updated long-only minimum-variance portfolio as of Dec 31, 2015,
    #### using again 10 years of historical returns for the covariance matrix estimation (Jan 2006 – Dec 2015).

    # Define the start and end date for the updated portfolio as of December 31, 2015
    update_start_date = '2006-01-01'
    update_end_date = '2015-12-31'

    # Filter the data for the 10-year period (January 2006 – December 2015)
    update_period_data = df[(df.index >= update_start_date) & (df.index <= update_end_date)]
    update_period_data.to_csv('2_WIP/update_period_data.csv', index=False)

    # Calculate the covariance matrix for this updated period
    update_covariance_matrix = update_period_data.cov()

    initial_weights_update = np.ones(len(update_period_data.columns)) / len(update_period_data.columns)
    objective = lambda weights: np.dot(weights.T, np.dot(update_covariance_matrix, weights))
    constraints_long_only_update = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Fully invested
        {'type': 'ineq', 'fun': lambda weights: weights}  # Long-only constraint
    )
    # Bounds for asset weights (0 to 1 for long-only)
    bounds = tuple((0, 1) for asset in range(len(update_period_data.columns)))
    # Perform portfolio optimization for the updated long-only minimum-variance portfolio
    min_var_portfolio_long_only_update = minimize(
        objective, initial_weights_update, method='SLSQP', bounds=bounds, constraints=constraints_long_only_update)


    # Calculate portfolio returns for the updated long-only minimum-variance portfolio as of Dec 31, 2015
    portfolio_weights_update = min_var_portfolio_long_only_update.x
    portfolio_returns_update = np.dot(update_period_data, portfolio_weights_update)
    total_weight_min_var_portfolio_long_only_update = round(min_var_portfolio_long_only_update.x.sum(), 1)
    # Print or store the results
    print("\nUpdated Long-Only Minimum-Variance Portfolio (as of Dec 31, 2015):")
    print("Industry:", industry_names)
    print("Weights:", min_var_portfolio_long_only_update.x)
    print("Total Weight:", total_weight_min_var_portfolio_long_only_update)
    print("Portfolio Variance:", min_var_portfolio_long_only_update.fun)

    results_min_var_portfolio_long_only_update = pd.DataFrame({'Industry': industry_names, 'Weights': min_var_portfolio_long_only_update.x})
    # Save the dataframes to Excel files
    results_min_var_portfolio_long_only_update.to_excel('2_WIP/results_min_var_portfolio_long_only_update.xlsx', index=False)


    ### get the realized volatility of this portfolio (portfolio_weights_update) for Jan 2016 through July 2023
    ### Calculate the annualized volatility for the updated long-only portfolio for Jan 2016 - Jul 2023
    out_of_sample_start_date_update = '2016-01-01'
    out_of_sample_end_date_update = '2023-07-31'
    out_of_sample_data_update = df[
        (df.index >= out_of_sample_start_date_update) & (df.index <= out_of_sample_end_date_update)]
    out_of_sample_portfolio_returns_update = np.dot(out_of_sample_data_update, portfolio_weights_update)
    out_of_sample_portfolio_volatility_update = annualized_volatility(out_of_sample_portfolio_returns_update)


    print("\nRealized Volatility for Jan 2016 - Jul 2023 (Updated Portfolio):",
          out_of_sample_portfolio_volatility_update)


    ###====================================================================================================
    ####======= 3E) performance expectations
    ### Focusing again on the long-only minimum-variance industry portfolio you created as-of Dec 31, 2004,
    ### plot the cumulative value of $1 invested in this strategy at its inception date (12/31/2004) through July 31, 2022 using monthly returns (so the final data point for July 2022 is the value of that initial investment at that date).
    ### Plot the value on a log-2 scale of the y-axis. In the same chart, add the cumulative value of $1 invested in the MSCI US index.


    # Step 1: Reading Data
    insample_data = pd.read_csv("2_WIP/in_sample_portfolio_returns_df.csv")
    outofsample_data = pd.read_csv("2_WIP/out_of_sample_portfolio_returns_df.csv")
    msci_data = pd.read_excel("1_Rawdata/output_MSCI_USA_index.xlsx")


    # Step 2: Data Preparation
    insample_data['Date'] = pd.to_datetime(insample_data['Date'])
    outofsample_data['Date'] = pd.to_datetime(outofsample_data['Date'])
    msci_data['Date'] = pd.to_datetime(msci_data['Date'])

    # Calculate cumulative returns for in-sample data
    cumulative_insample = (1 + insample_data['Minimum Variance Portfolio (Long-Only)']).cumprod()

    # Calculate cumulative returns for out-of-sample data
    cumulative_outofsample = (1 + outofsample_data['Minimum Variance Portfolio (Long-Only)']).cumprod()

    # Calculate cumulative returns for MSCI USA Index
    cumulative_msci = (1 + msci_data['monthly_return']).cumprod()

    # Plot the cumulative value of $1 invested in the strategy
    plt.figure(figsize=(12, 6))
    plt.plot(insample_data['Date'], cumulative_insample, label='In-Sample Portfolio_Strategy Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Value of $1 Invested')
    plt.title('Cumulative Value of $1 Invested in the Strategy [Minimum Variance Portfolio (Long-Only)]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("2_IMG/3E_Cumulative Value of $1 Invested_In-Sample.png")
    # plt.show()
    plt.pause(1)
    plt.close()

    # Plot the cumulative value of $1 invested in the strategy
    plt.figure(figsize=(12, 6))
    plt.plot(outofsample_data['Date'], cumulative_outofsample, label='Out-Of-Sample Portfolio_Strategy Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Value of $1 Invested')
    plt.title('Cumulative Value of $1 Invested in the Strategy [Minimum Variance Portfolio (Long-Only)]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("2_IMG/3E_Cumulative Value of $1 Invested_Out-Of-Sample.png")
    # plt.show()
    plt.pause(1)
    plt.close()

    ### combine 2 as one:Plot the cumulative value of $1 invested in the strategy
    plt.figure(figsize=(12, 6))
    plt.plot(insample_data['Date'], cumulative_insample, label='In-Sample Portfolio_Strategy Cumulative Returns',
             color='blue')
    plt.plot(outofsample_data['Date'], cumulative_outofsample,
             label='Out-Of-Sample Portfolio_Strategy Cumulative Returns', color='green')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Value of $1 Invested')
    plt.title('Cumulative Value of $1 Invested in the Strategy [Minimum Variance Portfolio (Long-Only)]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("2_IMG/3E_Cumulative Value of $1 Invested_2in1.png")
    # plt.show()
    plt.pause(1)
    plt.close()

    # Filter data based on the desired date range (start_date to end_date)
    start_date = '2004-12-31'
    end_date = '2022-07-31'
    # zoomed_in_data = insample_data[(insample_data['Date'] >= start_date) & (insample_data['Date'] <= end_date)]
    zoomed_in_data = outofsample_data[(outofsample_data['Date'] >= start_date) & (outofsample_data['Date'] <= end_date)]
    zoomed_in_msci = msci_data[(msci_data['Date'] >= start_date) & (msci_data['Date'] <= end_date)]

    # Calculate cumulative returns for the zoomed-in data
    cumulative_zoomed_in = (1 + zoomed_in_data['Minimum Variance Portfolio (Long-Only)']).cumprod()
    cumulative_zoomed_in_msci = (1 + zoomed_in_msci['monthly_return']).cumprod()

    # Plot the cumulative value of $1 invested in the strategy for the zoomed-in range
    if not zoomed_in_data.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(zoomed_in_data['Date'], cumulative_zoomed_in, label='Zoomed-In Portfolio_Strategy Cumulative Returns',
                 color='black')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Value of $1 Invested')
        plt.title('Cumulative Value of $1 Invested in the Strategy (Zoomed-In)')
        plt.legend()
        plt.grid()
        # Customize the date formatting to display both year and month
        date_format = mdates.DateFormatter("%Y-%m")
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("2_IMG/Zoomed-In Cumulative Value of $1 Invested.png")
        # plt.show()
        plt.pause(1)
        plt.close()
    else:
        print("No data available for the specified zoomed-in date range.")


    # Plot the cumulative value of $1 invested in the strategy for the zoomed-in range on a log-2 scale
    if not zoomed_in_data.empty:
        plt.figure(figsize=(12, 6))

        # Plot your portfolio returns on a log-2 scale
        plt.semilogy(zoomed_in_data['Date'], cumulative_zoomed_in, label='Zoomed-In Portfolio Cumulative Returns',
                     color='black')

        # Plot the cumulative value of $1 invested in the MSCI US index on a log-2 scale
        plt.semilogy(zoomed_in_msci['Date'], cumulative_zoomed_in_msci, label='MSCI US Index Cumulative Returns', color='blue')

        plt.xlabel('Date')
        plt.ylabel('Cumulative Value of $1 Invested (log-2 scale)')
        plt.title('Cumulative Value of $1 Invested (Log-2 Scale) in the Portfolio vs. MSCI US Index')
        plt.legend()
        plt.grid()

        # Customize the date formatting to display both year and month
        date_format = mdates.DateFormatter("%Y-%m")
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=90)
        step = 3 # Set the step value to control the number of labels displayed (e.g., show every 3rd label)
        x_ticks = plt.gca().get_xticks()
        plt.gca().set_xticks(x_ticks[::step])

        plt.tight_layout()
        plt.savefig("2_IMG/Log-2 Scale Cumulative Value vs. MSCI US Index.png")
        # plt.show()
        plt.pause(1)
        plt.close()
    else:
        print("No data available for the specified zoomed-in date range.")



    #### part 2: Digging deeper, identify the 10 industries that held up best (that is, had the highest cumulative return) during each of these two market crises (separately) and comment on whether these are traditional defensive industries or whether some at first sight are unexpected

    ### part 2.1: Return per industry:

    # Calculate portfolio returns per industry
    industry_names = out_of_sample_data.columns[1:]  # Exclude 'yearmonth' column
    portfolio_returns_per_industry = {}

    # Iterate through each industry and calculate portfolio returns
    for industry in industry_names:
        industry_returns = out_of_sample_data[industry].values
        industry_weight = min_var_portfolio_long_only.x[industry_names.get_loc(industry)]  # Get the weight for this industry
        industry_portfolio_return = industry_returns * industry_weight
        portfolio_returns_per_industry[industry] = np.sum(industry_portfolio_return)

    # Create a DataFrame with portfolio returns per industry
    portfolio_returns_per_industry_df = pd.DataFrame({
        'Industry': list(portfolio_returns_per_industry.keys()),
        'Portfolio Return': list(portfolio_returns_per_industry.values())
    })

    # Save the DataFrame to a CSV file
    portfolio_returns_per_industry_df.to_csv('2_WIP/outofsample_longonly_portfolio_returns_per_industry.csv', index=False)


    ### part 2.2: Return per industry per yearmonth:
    portfolio_returns_per_industry_per_yearmonth = {}
    portfolio_returns_per_industry_per_yearmonth['yearmonth'] = [date.strftime('%Y-%m') for date in out_of_sample_data.index]
    portfolio_returns_per_industry_per_yearmonth['yearmonth'] = pd.to_datetime(portfolio_returns_per_industry_per_yearmonth['yearmonth'], format='%Y-%m')

    # Iterate through each industry and calculate portfolio returns
    for industry in industry_names:
        industry_returns = out_of_sample_data[industry].values
        industry_weight = min_var_portfolio_long_only.x[industry_names.get_loc(industry)]  # Get the weight for this industry
        industry_portfolio_return = industry_returns * industry_weight
        portfolio_returns_per_industry_per_yearmonth[industry] = industry_portfolio_return

    # Create a DataFrame with portfolio returns per industry per year-month
    portfolio_returns_per_industry_per_yearmonth_df = pd.DataFrame(portfolio_returns_per_industry_per_yearmonth)

    # Save the DataFrame to a CSV file
    portfolio_returns_per_industry_per_yearmonth_df.to_csv('2_WIP/outofsample_longonly_portfolio_returns_per_industry_per_yearmonth.csv',
                                                           index=False)


    ### part 2.3: analysis
    # Sort the DataFrame by 'Portfolio Return' in descending order
    sorted_df = portfolio_returns_per_industry_df.sort_values(by='Portfolio Return', ascending=False)

    # Plot a bar chart for all industries
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_df['Industry'], sorted_df['Portfolio Return'], color='skyblue')
    plt.xlabel('Industry')
    plt.ylabel('Portfolio Return')
    plt.title('Portfolio Returns by Industry (Top 10 Highlighted)')
    plt.xticks(rotation=45)

    # Highlight the top 10 industries in green and add labels and return values
    highlight_color = 'green'
    for i in range(10):
        industry = sorted_df.iloc[i]['Industry']
        return_value = sorted_df.iloc[i]['Portfolio Return']
        plt.bar(industry, return_value, color=highlight_color, label=f'{industry}\nReturn: {return_value:.2f}')

    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig("2_IMG/top 10 industries in green_return values.png")
    # plt.show()
    plt.pause(1)
    plt.close()



    # Define the time periods for the market crises
    gfc_start_date = '2008-09-01'
    gfc_end_date = '2009-02-28'
    covid_start_date = '2020-02-01'
    covid_end_date = '2020-03-31'


    # Filter the data for the Global Financial Crisis
    gfc_data = portfolio_returns_per_industry_per_yearmonth_df[(portfolio_returns_per_industry_per_yearmonth_df['yearmonth'] >= gfc_start_date) & (portfolio_returns_per_industry_per_yearmonth_df['yearmonth'] <= gfc_end_date)]

    # Filter the data for the COVID market meltdown
    covid_data = portfolio_returns_per_industry_per_yearmonth_df[(portfolio_returns_per_industry_per_yearmonth_df['yearmonth'] >= covid_start_date) & (portfolio_returns_per_industry_per_yearmonth_df['yearmonth'] <= covid_end_date)]
    covid_data.to_csv('2_WIP/covid_data.csv')
    gfc_data.to_csv('2_WIP/gfc_data.csv')
    # Calculate cumulative returns for each industry
    gfc_data_cumulative_returns = gfc_data.drop('yearmonth', axis=1).cumsum()
    gfc_data_cumulative_returns.to_csv('2_WIP/gfc_data_cumulative_returns.csv')
    # Identify the top 10 industries with the highest cumulative returns
    top_10_industries = gfc_data_cumulative_returns.iloc[-1].nlargest(10)
    print('Cumulative Returns by Industry during Global Financial Crisis:')
    print(top_10_industries)

    plt.figure(figsize=(24, 12))
    for col in gfc_data_cumulative_returns.columns:
        # Use dashed lines for industries not in the top 10
        linestyle = '-' if col in top_10_industries.index else '--'
        plt.plot(gfc_data['yearmonth'], gfc_data_cumulative_returns[col], label=col, linestyle=linestyle)

    # Highlight the top 10 industries with solid lines
    for industry in top_10_industries.index:
        plt.plot(gfc_data['yearmonth'], gfc_data_cumulative_returns[industry], linewidth=3, label=industry,
                 linestyle='-')

    plt.title('Cumulative Returns by Industry during Global Financial Crisis')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Top 10 Industries')
    plt.grid()
    plt.savefig("2_IMG/Cumulative Returns by Industry during Global Financial Crisis.png")
    plt.pause(1)
    plt.close()


 # Calculate cumulative returns for each industry
    covid_data_cumulative_returns = covid_data.drop('yearmonth', axis=1).cumsum()
    covid_data_cumulative_returns.to_csv('2_WIP/covid_data_cumulative_returns.csv')

    # Identify the top 10 industries with the highest cumulative returns
    top_10_industries = covid_data_cumulative_returns.iloc[-1].nlargest(10)
    print('Cumulative Returns by Industry during COVID crisis:')
    print(top_10_industries)


    plt.figure(figsize=(24, 12))
    for col in covid_data_cumulative_returns.columns:
        # Use dashed lines for industries not in the top 10
        linestyle = '-' if col in top_10_industries.index else '--'
        plt.plot(covid_data['yearmonth'], covid_data_cumulative_returns[col], label=col, linestyle=linestyle)

    # Highlight the top 10 industries with solid lines
    for industry in top_10_industries.index:
        plt.plot(covid_data['yearmonth'], covid_data_cumulative_returns[industry], linewidth=3, label=industry,
                 linestyle='-')

    plt.title('Cumulative Returns by Industry during COVID crisis')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Top 10 Industries')
    plt.grid()
    plt.savefig("2_IMG/Cumulative Returns by Industry during COVID crisis.png")
    plt.pause(1)
    plt.close()








