import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
import yaml
import os
import logging

# --- New Imports for Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Matplotlib and Seaborn styles
plt.style.use('seaborn-v0_8')
sns.set_theme()

# --- Logging Configuration ---
logging.basicConfig(
    filename='stock_analysis.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- Configuration Loader ---

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file and replace environment variable placeholders.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            # Replace placeholders with actual environment variable values
            for section, params in config.items():
                if isinstance(params, dict):
                    for key, value in params.items():
                        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                            env_var = value[2:-1]
                            env_value = os.getenv(env_var)
                            if env_value is None:
                                raise ValueError(f"Environment variable '{env_var}' is not set.")
                            config[section][key] = env_value
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing the configuration file: {e}")


# --- Financial Ratios Comparison Code ---

def get_financial_data(ticker_list):
    """
    Retrieve key financial metrics for a list of tickers.
    """
    data_list = []

    for ticker in ticker_list:
        logging.info(f"Processing ticker: {ticker}")
        print(f"Processing {ticker}")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract necessary data
            data = {
                'Ticker': ticker,
                'Current Price': info.get('currentPrice', np.nan),
                'EPS (TTM)': info.get('trailingEps', np.nan),
                'EPS (Forward)': info.get('forwardEps', np.nan),
                'P/E Ratio': info.get('trailingPE', np.nan),
                'Forward P/E': info.get('forwardPE', np.nan),
                'Price/Book': info.get('priceToBook', np.nan),
                'Profit Margin': info.get('profitMargins', np.nan),
                'Return on Equity': info.get('returnOnEquity', np.nan),
                'Debt/Equity': info.get('debtToEquity', np.nan),
                'Market Cap': info.get('marketCap', np.nan)
            }

            # Calculate Earnings Growth Rate using historical net income
            financials = stock.financials
            if not financials.empty:
                possible_net_income_keys = ['Net Income', 'NetIncome', 'Net Income Common Stockholders']
                net_income = pd.Series(dtype=float)

                # Normalize index labels to lowercase for matching
                financials.index = financials.index.str.lower().str.strip()

                for key in possible_net_income_keys:
                    key_lower = key.lower().strip()
                    if key_lower in financials.index:
                        net_income = financials.loc[key_lower].dropna()
                        break

                if net_income.empty:
                    logging.warning(f"Net Income row not found for {ticker}. Available rows: {financials.index.tolist()}")
                    data['Earnings Growth Rate'] = np.nan
                else:
                    # Ensure the net income data is sorted by date
                    net_income = net_income.sort_index()
                    years = len(net_income)
                    if years >= 2:
                        start_income = net_income.iloc[0]
                        end_income = net_income.iloc[-1]
                        periods = years - 1
                        if start_income > 0 and end_income > 0:
                            cagr = ((end_income / start_income) ** (1 / periods) - 1) * 100
                            data['Earnings Growth Rate'] = cagr
                        else:
                            data['Earnings Growth Rate'] = np.nan
                    else:
                        data['Earnings Growth Rate'] = np.nan
            else:
                logging.warning(f"No financial data available for {ticker}.")
                data['Earnings Growth Rate'] = np.nan

            # Calculate PEG Ratio if possible
            if pd.notna(data['P/E Ratio']) and pd.notna(data['Earnings Growth Rate']) and data['Earnings Growth Rate'] != 0:
                data['PEG Ratio'] = data['P/E Ratio'] / data['Earnings Growth Rate']
            else:
                data['PEG Ratio'] = np.nan

            data_list.append(data)

        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            print(f"Error processing {ticker}: {e}")
            continue

    df = pd.DataFrame(data_list)
    return df

def perform_valuation(df, focused_ticker):
    """
    Compare the focused ticker's financial metrics against its peers.
    """
    # Focus on the specified ticker
    focused_data = df[df['Ticker'] == focused_ticker]
    if focused_data.empty:
        logging.error(f"{focused_ticker} data not found in the DataFrame.")
        print(f"{focused_ticker} data not found in the DataFrame.")
        return pd.DataFrame()
    focused_data = focused_data.iloc[0]

    # Compare with peers
    peers = df[df['Ticker'] != focused_ticker]
    if peers.empty:
        logging.error("No peer data available.")
        print("No peer data available.")
        return pd.DataFrame()
    peer_means = peers.mean(numeric_only=True)

    valuation = {}
    metrics = ['P/E Ratio', 'PEG Ratio', 'Price/Book', 'Profit Margin', 'Return on Equity', 'Debt/Equity']
    for metric in metrics:
        focused_value = focused_data.get(metric, np.nan)
        peer_value = peer_means.get(metric, np.nan)
        if pd.notna(focused_value) and pd.notna(peer_value):
            difference = focused_value - peer_value
            percent_diff = (difference / peer_value) * 100 if peer_value != 0 else np.nan
        else:
            difference = np.nan
            percent_diff = np.nan
        valuation[metric] = {
            focused_ticker: focused_value,
            'Peers Average': peer_value,
            'Difference': difference,
            'Percent Difference (%)': percent_diff
        }

    valuation_df = pd.DataFrame(valuation).T
    return valuation_df

# --- DCF Valuation Code (Enhanced) ---

class DCFValuation:
    def __init__(self, ticker, forecast_years=5, perpetual_growth_rate=3, fred_api_key=None, config=None):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.forecast_years = forecast_years
        self.perpetual_growth_rate = perpetual_growth_rate
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        self.config = config  # Store the config for default values

    def _get_financial_item(self, df, possible_keys):
        """
        Helper method to retrieve financial statement items with flexible key names.
        """
        df.index = df.index.str.lower().str.strip()  # Normalize index labels
        for key in possible_keys:
            key_normalized = key.lower().strip()
            if key_normalized in df.index:
                return df.loc[key_normalized].dropna()
        return pd.Series(dtype=float)

    def calculate_capex(self, cf):
        """
        Calculate CapEx by taking the difference in Property, Plant & Equipment (PPE) from the balance sheet.
        """
        try:
            balance = self.stock.balance_sheet
            ppe = self._get_financial_item(balance, ['property plant equipment', 'property, plant & equipment net']).astype(float)
            if ppe.empty:
                logging.warning(f"PPE not found for {self.ticker}. Cannot calculate CapEx.")
                return pd.Series(0, index=cf.columns)
            # CapEx = Current PPE - Previous PPE + Depreciation
            depreciation = self._get_financial_item(cf, ['depreciation', 'depreciation & amortization', 'depreciation amortization depletion']).astype(float)
            if depreciation.empty:
                depreciation = pd.Series(0, index=cf.columns)
                logging.warning(f"Depreciation not found for {self.ticker}. Assuming 0.")
            capex = ppe.diff(periods=-1) + depreciation
            capex = capex.dropna()
            return capex
        except Exception as e:
            logging.error(f"Error calculating CapEx: {e}")
            return pd.Series(0, index=cf.columns)

    def estimate_depreciation(self, income):
        """
        Estimate Depreciation as a percentage of Revenue if not directly available.
        """
        try:
            revenue = self._get_financial_item(income, ['total revenue', 'revenue']).astype(float)
            if revenue.empty:
                logging.warning(f"Revenue not found for {self.ticker}. Cannot estimate Depreciation.")
                return pd.Series(0, index=income.columns)
            # Assume average Depreciation rate over Revenue
            depreciation_rate = 0.05  # Assuming Depreciation is 5% of Revenue
            depreciation = revenue * depreciation_rate
            return depreciation.mean()
        except Exception as e:
            logging.error(f"Error estimating Depreciation: {e}")
            return 0

    def get_historical_financials(self):
        """
        Extract historical financial data from yfinance.
        Returns a dictionary containing FCF, Revenue, EBIT, Net Income, Total Debt, Cash and Equivalents, Revenue Growth, EBIT Margin, Tax Rate, Depreciation, CapEx Percent, Average NWC Change.
        """
        try:
            # Get financial statements
            cf = self.stock.cashflow
            income = self.stock.financials
            balance = self.stock.balance_sheet

            # Revenue, EBIT, Net Income
            revenue = self._get_financial_item(income, ['Total Revenue', 'Revenue']).astype(float)
            ebit = self._get_financial_item(income, ['Ebit', 'EBIT', 'Operating Income', 'Operating Income or Loss']).astype(float)
            net_income = self._get_financial_item(income, ['Net Income', 'NetIncome', 'Net Income Common Stockholders']).astype(float)

            # Check for empty data
            if revenue.empty:
                logging.warning(f"Revenue data missing for {self.ticker}.")
            if ebit.empty:
                logging.warning(f"EBIT data missing for {self.ticker}.")

            # Calculate EBIT Margin
            if not ebit.empty and not revenue.empty:
                ebit_margin = ebit / revenue
                ebit_margin = ebit_margin.dropna()
                average_ebit_margin = ebit_margin.mean()
            else:
                logging.warning(f"EBIT or Revenue data missing for {self.ticker}. Cannot calculate EBIT Margin.")
                # Retrieve default from config
                default_ebit_margin = self.config.get('src', {}).get('default_ebit_margin', 0.15)
                average_ebit_margin = default_ebit_margin
                logging.info(f"Using default EBIT Margin from config: {average_ebit_margin}")

            # Effective Tax Rate
            tax_rate = self._get_effective_tax_rate()
            if tax_rate is None:
                # Retrieve default from config
                default_tax_rate = self.config.get('src', {}).get('default_tax_rate', 0.21)
                tax_rate = default_tax_rate
                logging.info(f"Using default Tax Rate from config: {tax_rate}")

            # Get balance sheet items
            current_assets = self._get_financial_item(balance, ['Total Current Assets', 'Current Assets']).astype(float)
            current_liabilities = self._get_financial_item(balance, ['Total Current Liabilities', 'Current Liabilities']).astype(float)

            # Calculate NWC Changes
            if not current_assets.empty and not current_liabilities.empty:
                nwc = current_assets - current_liabilities
                nwc_changes = nwc.diff(periods=-1)
                average_nwc_change = nwc_changes.mean()
            else:
                logging.warning(f"Current Assets or Current Liabilities data missing for {self.ticker}. Cannot calculate NWC changes.")
                average_nwc_change = 0  # Default to zero

            # Capital Expenditures
            capital_expenditure = self._get_financial_item(cf, ['Capital Expenditures', 'Capital Expenditure', 'Investment in Property, Plant and Equipment', 'Capital Expenditures (Cash Flow Statement)']).astype(float)
            if capital_expenditure.empty:
                logging.warning(f"Capital Expenditures not found for {self.ticker}. Calculating CapEx.")
                capital_expenditure = self.calculate_capex(cf)

            # Operating Cash Flow
            operating_cash_flow = self._get_financial_item(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities', 'Net Cash Provided by Operating Activities']).astype(float)
            if operating_cash_flow.empty:
                logging.warning(f"Operating Cash Flow not found for {self.ticker}. Assuming 0.")
                operating_cash_flow = pd.Series(0, index=capital_expenditure.index)
            else:
                # Ensure "Operating Cash Flow" is correctly captured
                logging.info(f"Operating Cash Flow for {self.ticker}: {operating_cash_flow.tolist()}")

            # Free Cash Flow
            free_cash_flow = operating_cash_flow + capital_expenditure  # CapEx is negative in cash flow statements

            # Depreciation
            depreciation = self._get_financial_item(cf, ['Depreciation', 'Depreciation & Amortization', 'Depreciation Amortization Depletion']).astype(float)
            if depreciation.empty:
                logging.warning(f"Depreciation not found for {self.ticker}. Estimating Depreciation.")
                depreciation = self.estimate_depreciation(income)
            else:
                depreciation = depreciation.mean()

            # Total Debt, Cash and Equivalents
            total_debt = self._get_financial_item(balance, ['Total Debt', 'Long Term Debt']).astype(float)
            if total_debt.empty:
                logging.warning(f"Total Debt not found for {self.ticker}. Assuming 0.")
                total_debt = pd.Series(0, index=cf.columns)
            cash_and_equivalents = self._get_financial_item(balance, ['Cash', 'Cash And Cash Equivalents']).astype(float)
            if cash_and_equivalents.empty:
                logging.warning(f"Cash and Equivalents not found for {self.ticker}. Assuming 0.")
                cash_and_equivalents = pd.Series(0, index=cf.columns)

            # CapEx as % of Revenue
            if not capital_expenditure.empty and not revenue.empty:
                capex_percent = capital_expenditure / revenue
                average_capex_percent = capex_percent.mean()
            else:
                logging.warning(f"Capital Expenditure or Revenue data missing for {self.ticker}. Cannot calculate CapEx percentage.")
                average_capex_percent = 0.05  # Default 5%

            historical_data = {
                'Free Cash Flow': free_cash_flow,
                'Revenue': revenue,
                'EBIT': ebit,
                'Net Income': net_income,
                'EBIT Margin': average_ebit_margin,
                'Tax Rate': tax_rate,
                'Average NWC Change': average_nwc_change,
                'Depreciation': depreciation,
                'CapEx Percent': average_capex_percent
            }

            return historical_data

        except Exception as e:
            logging.error(f"Error getting historical financials: {e}")
            print(f"Error getting historical financials: {e}")
            return None

    def calculate_cost_of_equity(self):
        """
        Calculate the cost of equity using the Capital Asset Pricing Model (CAPM).
        """
        try:
            risk_free_rate = self.get_risk_free_rate() / 100  # Convert percentage to decimal
            beta = self.stock.info.get('beta', 1.0)
            market_risk_premium = 0.06  # Assumed market risk premium of 6%
            cost_of_equity = risk_free_rate + beta * market_risk_premium
            logging.info(f"Calculated Cost of Equity: {cost_of_equity:.2%}")
            return cost_of_equity
        except Exception as e:
            logging.error(f"Error calculating cost of equity: {e}")
            print(f"Error calculating cost of equity: {e}")
            return None

    def calculate_wacc(self):
        """
        Calculate the Weighted Average Cost of Capital (WACC).
        """
        try:
            cost_of_equity = self.calculate_cost_of_equity()
            if cost_of_equity is None:
                logging.error("Cost of equity could not be calculated.")
                return None

            info = self.stock.info

            # Cost of Debt
            total_debt = info.get('totalDebt', 0)
            if not self.stock.financials.empty:
                interest_expense_series = self._get_financial_item(self.stock.financials, ['interest expense', 'interest expense, net']).astype(float)
                interest_expense = interest_expense_series.iloc[0] if not interest_expense_series.empty else 0
            else:
                interest_expense = 0

            if total_debt > 0:
                effective_tax_rate = self._get_effective_tax_rate()
                if effective_tax_rate is None:
                    # Retrieve default from config
                    default_tax_rate = self.config.get('src', {}).get('default_tax_rate', 0.21)
                    effective_tax_rate = default_tax_rate
                    logging.info(f"Using default Tax Rate from config for WACC calculation: {effective_tax_rate}")
                cost_of_debt = (abs(interest_expense) / total_debt) * (1 - effective_tax_rate)
            else:
                cost_of_debt = 0

            # Capital structure
            market_cap = info.get('marketCap', 0)
            total_capital = market_cap + total_debt
            if total_capital == 0:
                logging.error("Total capital is zero, cannot calculate WACC.")
                print("Total capital is zero, cannot calculate WACC.")
                return None

            equity_weight = market_cap / total_capital
            debt_weight = total_debt / total_capital

            wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt)
            logging.info(f"Calculated WACC: {wacc:.2%}")
            return wacc
        except Exception as e:
            logging.error(f"Error calculating WACC: {e}")
            print(f"Error calculating WACC: {e}")
            return None

    def _get_effective_tax_rate(self):
        """
        Retrieve the effective tax rate from financials.
        """
        try:
            income = self.stock.financials
            tax_expense = self._get_financial_item(income, ['tax provision', 'income tax expense', 'provision for income taxes']).astype(float)
            pretax_income = self._get_financial_item(income, ['pretax income', 'income before tax', 'earnings before tax']).astype(float)

            if not tax_expense.empty and not pretax_income.empty:
                tax_rates = tax_expense / pretax_income
                effective_tax_rate = tax_rates.mean()
                logging.info(f"Calculated Effective Tax Rate: {effective_tax_rate:.2%}")
                return effective_tax_rate
            else:
                logging.warning("Tax Expense or Pretax Income data missing.")
                return None
        except Exception as e:
            logging.error(f"Error retrieving effective tax rate: {e}")
            print(f"Error retrieving effective tax rate: {e}")
            return None

    def get_risk_free_rate(self):
        """
        Retrieve the current 10-year Treasury yield (risk-free rate) from FRED.
        If FRED API is unavailable or fails, fallback to yfinance.
        If both sources fail, return a default risk-free rate.

        Returns:
            float: The risk-free rate as a percentage.
        """
        # Attempt to fetch from FRED if the API key is provided
        if self.fred:
            try:
                # Define the date range for recent data (e.g., last 30 days)
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

                # Fetch the 10-year Treasury yield series (DGS10)
                treasury_data = self.fred.get_series(
                    'DGS10',
                    observation_start=start_date,
                    observation_end=end_date
                )

                # Drop any NaN values and get the most recent rate
                latest_rate = treasury_data.dropna().iloc[-1]
                logging.info(f"Fetched Risk-Free Rate from FRED: {latest_rate:.2f}%")
                return latest_rate

            except Exception as e:
                logging.error(f"Error fetching Risk-Free Rate from FRED: {e}")
                print(f"Error fetching Risk-Free Rate from FRED: {e}")

        # Fallback to yfinance if FRED is unavailable or failed
        try:
            treasury = yf.Ticker('^TNX')
            rate = treasury.info.get('regularMarketPrice', 3.5)  # Default to 3.5% if not available
            logging.info(f"Fetched Risk-Free Rate from yfinance (^TNX): {rate}%")
            return rate

        except Exception as e:
            logging.error(f"Error fetching Risk-Free Rate from yfinance: {e}")
            print(f"Error fetching Risk-Free Rate from yfinance: {e}")

        # Final fallback to a default value if both sources fail
        default_rate = 3.5  # You can adjust this default rate as needed
        logging.warning(f"Both FRED and yfinance failed. Using default Risk-Free Rate: {default_rate}%")
        print(f"Both FRED and yfinance failed. Using default Risk-Free Rate: {default_rate}%")
        return default_rate

    def forecast_free_cash_flows(self, historical_data, assumptions):
        """
        Forecast future Free Cash Flows based on historical data and assumptions.
        """
        try:
            # Unpack historical data
            revenue = historical_data['Revenue']
            ebit_margin = assumptions.get('ebit_margin', historical_data['EBIT Margin'])
            tax_rate = assumptions.get('tax_rate', historical_data['Tax Rate'])
            capex_percent = historical_data.get('CapEx Percent', 0.05)
            nwc_change = historical_data.get('Average NWC Change', 0)

            # Validate ebit_margin
            if pd.isna(ebit_margin) or ebit_margin == 0:
                logging.warning("EBIT Margin could not be calculated.")
                # Retrieve default from config
                default_ebit_margin = self.config.get('src', {}).get('default_ebit_margin', 0.15)
                ebit_margin = default_ebit_margin
                logging.info(f"Using default EBIT Margin from config: {ebit_margin}")

            # Validate tax_rate
            if pd.isna(tax_rate) or tax_rate == 0:
                logging.warning("Tax Rate could not be calculated.")
                # Retrieve default from config
                default_tax_rate = self.config.get('src', {}).get('default_tax_rate', 0.21)
                tax_rate = default_tax_rate
                logging.info(f"Using default Tax Rate from config: {tax_rate}")

            # Validate capex_percent
            if pd.isna(capex_percent) or capex_percent == 0:
                logging.warning("CapEx percentage could not be calculated. Using default of 5%.")
                capex_percent = 0.05  # Default value

            # Validate nwc_change
            if pd.isna(nwc_change):
                logging.warning("NWC change could not be calculated. Using default of 0.")
                nwc_change = 0

            depreciation = historical_data.get('Depreciation', 0)
            if pd.isna(depreciation):
                depreciation = 0
                logging.warning(f"Depreciation not available. Using default value: {depreciation}")

            # Forecast parameters
            forecast_years = self.forecast_years
            revenue_growth_rate = assumptions.get('revenue_growth_rate', 0.05)

            forecast_df = pd.DataFrame()
            forecast_df['Year'] = [f'Year {i}' for i in range(1, forecast_years + 1)]

            # Starting values
            last_revenue = revenue.iloc[-1]
            forecast_revenue = []
            forecast_ebit = []
            forecast_tax = []
            forecast_nopat = []
            forecast_capex = []
            forecast_depreciation = []
            forecast_nwc = []
            forecast_fcf = []

            for i in range(forecast_years):
                # Revenue forecast
                rev = last_revenue * (1 + revenue_growth_rate) ** (i + 1)
                forecast_revenue.append(rev)

                # EBIT forecast
                ebit = rev * ebit_margin
                forecast_ebit.append(ebit)

                # Tax
                tax = ebit * tax_rate
                forecast_tax.append(tax)

                # NOPAT (Net Operating Profit After Tax)
                nopat = ebit - tax
                forecast_nopat.append(nopat)

                # CapEx forecast
                capex_forecast = rev * capex_percent
                forecast_capex.append(capex_forecast)

                # NWC Changes
                nwc_forecast = nwc_change  # Could be adjusted based on revenue growth
                forecast_nwc.append(nwc_forecast)

                # Depreciation forecast (assuming it remains constant)
                dep_forecast = depreciation
                forecast_depreciation.append(dep_forecast)

                # Free Cash Flow
                fcf = nopat + dep_forecast - capex_forecast - nwc_forecast
                forecast_fcf.append(fcf)

            forecast_df['Revenue'] = forecast_revenue
            forecast_df['EBIT'] = forecast_ebit
            forecast_df['Tax'] = forecast_tax
            forecast_df['NOPAT'] = forecast_nopat
            forecast_df['CapEx'] = forecast_capex
            forecast_df['Depreciation'] = forecast_depreciation
            forecast_df['Change in NWC'] = forecast_nwc
            forecast_df['Free Cash Flow'] = forecast_fcf

            logging.info("Forecasted Free Cash Flows:")
            logging.info(forecast_df[['Year', 'Free Cash Flow']])

            return forecast_df
        except Exception as e:
            logging.error(f"Error forecasting free cash flows: {e}")
            print(f"Error forecasting free cash flows: {e}")
            return pd.DataFrame()

    def calculate_terminal_value(self, last_nopat, wacc, terminal_growth_rate=None):
        """
        Calculate terminal value using the perpetuity growth method.
        """
        try:
            if terminal_growth_rate is None:
                terminal_growth_rate = self.perpetual_growth_rate / 100  # Default to instance's perpetual growth rate
            else:
                terminal_growth_rate = terminal_growth_rate / 100      # Convert percentage to decimal

            if (wacc - terminal_growth_rate) <= 0:
                logging.error("Invalid WACC and terminal growth rate combination for terminal value calculation.")
                print("Invalid WACC and terminal growth rate combination for terminal value calculation.")
                return None

            terminal_value = last_nopat * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
            logging.info(f"Calculated Terminal Value with TGR {terminal_growth_rate*100}%: {terminal_value}")
            return terminal_value
        except Exception as e:
            logging.error(f"Error calculating terminal value: {e}")
            print(f"Error calculating terminal value: {e}")
            return None

    def calculate_present_value(self, forecast_df, terminal_value, wacc):
        """
        Calculate the present value of forecasted FCFs and terminal value.
        """
        try:
            cash_flows = forecast_df['Free Cash Flow'].values
            present_values = []
            for i, cf in enumerate(cash_flows):
                pv = cf / (1 + wacc) ** (i + 1)
                present_values.append(pv)
            # Discount terminal value
            pv_terminal = terminal_value / (1 + wacc) ** len(cash_flows)
            total_pv = sum(present_values) + pv_terminal
            logging.info(f"Total Present Value: {total_pv}")
            return total_pv
        except Exception as e:
            logging.error(f"Error calculating present value: {e}")
            print(f"Error calculating present value: {e}")
            return None

    def calculate_intrinsic_value(self, forecast_df, total_pv):
        """
        Calculate intrinsic value per share by adjusting for cash and debt.
        """
        try:
            info = self.stock.info
            shares_outstanding = info.get('sharesOutstanding', 0)
            total_debt = info.get('totalDebt', 0)
            cash_and_equivalents = info.get('totalCash', 0)

            # Equity Value = Total PV + Cash - Debt
            equity_value = total_pv + cash_and_equivalents - total_debt
            logging.info(f"Equity Value: {equity_value}")

            if shares_outstanding > 0:
                intrinsic_value = equity_value / shares_outstanding
                current_price = info.get('currentPrice', 0)
                if current_price > 0:
                    upside_downside = ((intrinsic_value / current_price) - 1) * 100
                else:
                    upside_downside = np.nan
                valuation = {
                    'Intrinsic Value per Share': intrinsic_value,
                    'Current Price': current_price,
                    'Upside/Downside %': upside_downside
                }
                logging.info(f"Intrinsic Value per Share: {intrinsic_value}")
                logging.info(f"Current Price: {current_price}")
                logging.info(f"Upside/Downside %: {upside_downside}")
                return valuation
            else:
                logging.error("Shares outstanding not found or zero.")
                print("Shares outstanding not found or zero.")
                return None
        except Exception as e:
            logging.error(f"Error calculating intrinsic value: {e}")
            print(f"Error calculating intrinsic value: {e}")
            return None

    def sensitivity_analysis(self, historical_data, wacc, revenue_growth_rates, terminal_growth_rates):
        """
        Perform sensitivity analysis on revenue growth rate and terminal growth rate.
        """
        try:
            sensitivity_results = []

            for gr in revenue_growth_rates:
                for tgr in terminal_growth_rates:
                    # Create a copy of assumptions with current growth rates
                    assumptions = {
                        'revenue_growth_rate': gr,
                        'ebit_margin': self.config.get('src', {}).get('ebit_margin', None),      # Use config default or None
                        'tax_rate': self.config.get('src', {}).get('tax_rate', None),            # Use config default or None
                        'capex_percent': None,    # Use historical or default
                        'nwc_change': None        # Use historical or default
                    }

                    # Forecast Free Cash Flows with current assumptions
                    forecast_df = self.forecast_free_cash_flows(historical_data, assumptions)
                    if forecast_df.empty:
                        logging.warning(f"Forecasting failed for Revenue Growth Rate: {gr}, Terminal Growth Rate: {tgr}. Skipping.")
                        continue

                    # Calculate Terminal Value with current terminal growth rate
                    last_nopat = forecast_df['NOPAT'].iloc[-1]
                    terminal_value = self.calculate_terminal_value(last_nopat, wacc, terminal_growth_rate=tgr)
                    if terminal_value is None:
                        logging.warning(f"Terminal Value calculation failed for Revenue Growth Rate: {gr}, Terminal Growth Rate: {tgr}. Skipping.")
                        continue

                    # Calculate Present Value
                    total_pv = self.calculate_present_value(forecast_df, terminal_value, wacc)
                    if total_pv is None:
                        logging.warning(f"Present Value calculation failed for Revenue Growth Rate: {gr}, Terminal Growth Rate: {tgr}. Skipping.")
                        continue

                    # Calculate Intrinsic Value
                    valuation = self.calculate_intrinsic_value(forecast_df, total_pv)
                    if valuation is None:
                        logging.warning(f"Intrinsic Value calculation failed for Revenue Growth Rate: {gr}, Terminal Growth Rate: {tgr}. Skipping.")
                        continue

                    # Append results
                    sensitivity_results.append({
                        'Revenue Growth Rate (%)': gr * 100,
                        'Terminal Growth Rate (%)': tgr * 100,
                        'Intrinsic Value per Share': valuation['Intrinsic Value per Share']
                    })

            sensitivity_df = pd.DataFrame(sensitivity_results)
            logging.info("Sensitivity Analysis Completed.")
            logging.info(sensitivity_df)
            return sensitivity_df

        except Exception as e:
            logging.error(f"Error during sensitivity analysis: {e}")
            print(f"Error during sensitivity analysis: {e}")
            return pd.DataFrame()

    def run_dcf_analysis(self, historical_data, assumptions, perform_sensitivity=False, revenue_growth_rates=None, terminal_growth_rates=None):
        """
        Run complete DCF analysis for the given historical data and assumptions.
        Optionally perform sensitivity analysis.
        """
        try:
            # Calculate WACC
            wacc = self.calculate_wacc()
            if wacc is None:
                print("WACC could not be calculated.")
                return None, None, None

            # Forecast Free Cash Flows
            forecast_df = self.forecast_free_cash_flows(historical_data, assumptions)
            if forecast_df.empty:
                print("Failed to forecast free cash flows.")
                return None, None, None

            # Calculate Terminal Value
            last_nopat = forecast_df['NOPAT'].iloc[-1]
            terminal_value = self.calculate_terminal_value(last_nopat, wacc)
            if terminal_value is None:
                print("Failed to calculate terminal value.")
                return None, None, None

            # Calculate Present Value
            total_pv = self.calculate_present_value(forecast_df, terminal_value, wacc)
            if total_pv is None:
                print("Failed to calculate present value.")
                return None, None, None

            # Calculate Intrinsic Value
            valuation = self.calculate_intrinsic_value(forecast_df, total_pv)
            if valuation:
                print("\nValuation Results:")
                for metric, value in valuation.items():
                    if 'Price' in metric or 'Value' in metric:
                        print(f"{metric}: ${value:,.2f}")
                    else:
                        print(f"{metric}: {value:.2f}%")
            else:
                print("\nValuation could not be calculated.")

            # Perform Sensitivity Analysis if requested
            sensitivity_df = None
            if perform_sensitivity and revenue_growth_rates and terminal_growth_rates:
                sensitivity_df = self.sensitivity_analysis(historical_data, wacc, revenue_growth_rates, terminal_growth_rates)
                if not sensitivity_df.empty:
                    print("\nSensitivity Analysis Results:")
                    print(sensitivity_df)
                    # Save to CSV
                    sensitivity_df.to_csv('sensitivity_analysis.csv', index=False)
                    logging.info("Sensitivity analysis results saved to 'sensitivity_analysis.csv'.")
                    print("\nSensitivity analysis results saved to 'sensitivity_analysis.csv'.")
                else:
                    print("\nSensitivity analysis could not be performed.")
            return valuation, sensitivity_df, forecast_df

        except Exception as e:
            logging.error(f"Error running DCF analysis: {e}")
            print(f"Error running DCF analysis: {e}")
            return None, None, None

# --- Summary Insights Function ---

def generate_summary(valuation_df, dcf_valuation, sensitivity_df=None):
    """
    Generate a textual summary based on src comparisons and DCF results.
    Optionally include sensitivity analysis insights.
    """
    print("\n--- Summary Insights ---\n")

    # Valuation Comparison Insights
    print("Valuation Comparison Insights:")
    for metric in valuation_df.index:
        focused_val = valuation_df.loc[metric, 'MA'] if 'MA' in valuation_df.columns else valuation_df.loc[metric, 'focused_ticker']
        peer_avg = valuation_df.loc[metric, 'Peers Average']
        diff = valuation_df.loc[metric, 'Difference']
        percent_diff = valuation_df.loc[metric, 'Percent Difference (%)']

        if metric == 'PEG Ratio':
            if percent_diff < 0:
                interpretation = "undervalued considering growth."
            elif percent_diff > 0:
                interpretation = "overvalued considering growth."
            else:
                interpretation = "fairly valued considering growth."
        elif metric in ['P/E Ratio', 'Price/Book']:
            if percent_diff > 0:
                interpretation = "higher than peers, possibly overvalued."
            else:
                interpretation = "lower than peers, possibly undervalued."
        elif metric in ['Profit Margin', 'Return on Equity']:
            interpretation = "better than peers, indicating superior performance."
        elif metric == 'Debt/Equity':
            if percent_diff > 0:
                interpretation = "more leveraged than peers, increasing financial risk."
            else:
                interpretation = "less leveraged than peers, indicating lower financial risk."
        else:
            interpretation = "No interpretation available."

        print(f"- {metric}: {focused_val} vs {peer_avg:.2f} (Difference: {diff:.2f}, {percent_diff:.2f}%) -> {interpretation}")

    # DCF Valuation Insights
    print("\nDCF Valuation Insights:")
    if dcf_valuation:
        intrinsic = dcf_valuation.get('Intrinsic Value per Share', np.nan)
        current = dcf_valuation.get('Current Price', np.nan)
        upside_downside = dcf_valuation.get('Upside/Downside %', np.nan)

        if upside_downside < 0:
            print(f"- The intrinsic value per share (${intrinsic:,.2f}) is significantly lower than the current price (${current:,.2f}), indicating potential overvaluation.")
        elif upside_downside > 0:
            print(f"- The intrinsic value per share (${intrinsic:,.2f}) is higher than the current price (${current:,.2f}), suggesting potential undervaluation.")
        else:
            print("- The intrinsic value per share is equal to the current price, indicating fair src.")
    else:
        print("- DCF src could not be performed.")

    # Sensitivity Analysis Insights
    if sensitivity_df is not None and not sensitivity_df.empty:
        print("\nSensitivity Analysis Insights:")
        print("The intrinsic value per share varies based on different Revenue Growth Rates and Terminal Growth Rates.")
        print("Refer to 'sensitivity_analysis.csv' for detailed results.")
    else:
        print("\nNo Sensitivity Analysis performed.")

    print("\n--- End of Summary ---\n")

# --- Visualization Functions ---

def plot_forecasted_fcf(forecast_df, ticker):
    """
    Plot the forecasted Free Cash Flows over the forecast period.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_df['Year'], forecast_df['Free Cash Flow'] / 1e6, marker='o', linestyle='-')
    plt.title(f'Forecasted Free Cash Flows for {ticker}')
    plt.xlabel('Year')
    plt.ylabel('Free Cash Flow (in millions USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{ticker}_forecasted_fcf.png')
    plt.close()
    logging.info(f"Forecasted Free Cash Flows plot saved as '{ticker}_forecasted_fcf.png'.")

def plot_forecast_components(forecast_df, ticker):
    """
    Plot the components of the forecast over the forecast period.
    """
    components = ['Revenue', 'EBIT', 'NOPAT', 'CapEx', 'Depreciation']
    forecast_df_plot = forecast_df.copy()
    forecast_df_plot[components] = forecast_df_plot[components] / 1e6  # Convert to millions

    forecast_df_plot.set_index('Year', inplace=True)
    forecast_df_plot[components].plot(kind='bar', figsize=(12, 8))
    plt.title(f'Forecast Components for {ticker}')
    plt.xlabel('Year')
    plt.ylabel('Amount (in millions USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{ticker}_forecast_components.png')
    plt.close()
    logging.info(f"Forecast components plot saved as '{ticker}_forecast_components.png'.")

def plot_sensitivity_analysis(sensitivity_df, ticker):
    """
    Plot a heatmap of the sensitivity analysis results.
    """
    try:
        pivot_table = sensitivity_df.pivot(index="Revenue Growth Rate (%)", columns="Terminal Growth Rate (%)", values="Intrinsic Value per Share")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(f"Sensitivity Analysis: Intrinsic Value per Share for {ticker}")
        plt.ylabel("Revenue Growth Rate (%)")
        plt.xlabel("Terminal Growth Rate (%)")
        plt.tight_layout()
        plt.savefig(f'{ticker}_sensitivity_analysis.png')
        plt.close()
        logging.info(f"Sensitivity analysis heatmap saved as '{ticker}_sensitivity_analysis.png'.")
    except Exception as e:
        logging.error(f"Error plotting sensitivity analysis heatmap: {e}")
        print(f"Error plotting sensitivity analysis heatmap: {e}")

def plot_financial_metrics_comparison(df, focused_ticker):
    """
    Plot bar charts comparing financial metrics of the focused ticker against peers.
    """
    metrics = ['P/E Ratio', 'PEG Ratio', 'Price/Book', 'Profit Margin', 'Return on Equity', 'Debt/Equity']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        data = df[['Ticker', metric]].dropna()
        sns.barplot(x='Ticker', y=metric, data=data)
        plt.title(f'{metric} Comparison')
        plt.xlabel('Ticker')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f'{metric.lower().replace("/", "_")}_comparison.png')
        plt.close()
        logging.info(f"{metric} comparison plot saved as '{metric.lower().replace('/', '_')}_comparison.png'.")

def plot_pe_vs_growth(df):
    """
    Plot a scatter plot of P/E Ratio vs Earnings Growth Rate.
    """
    plt.figure(figsize=(10, 6))
    data = df[['Ticker', 'P/E Ratio', 'Earnings Growth Rate']].dropna()
    sns.scatterplot(x='Earnings Growth Rate', y='P/E Ratio', data=data, hue='Ticker', s=100)
    for i in range(data.shape[0]):
        plt.text(x=data['Earnings Growth Rate'].iloc[i]+0.5, y=data['P/E Ratio'].iloc[i]+0.5, s=data['Ticker'].iloc[i])
    plt.title('P/E Ratio vs Earnings Growth Rate')
    plt.xlabel('Earnings Growth Rate (%)')
    plt.ylabel('P/E Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pe_vs_growth.png')
    plt.close()
    logging.info("P/E Ratio vs Earnings Growth Rate scatter plot saved as 'pe_vs_growth.png'.")

# --- Main Execution ---

def main():
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        print(f"Failed to load configuration: {e}")
        return

    fred_api_key = config.get('fred', {}).get('api_key', None)

    if fred_api_key is None:
        logging.error("FRED API key is not provided. Please set it in the 'config.yaml' or as an environment variable.")
        print("FRED API key is not provided. Please set it in the 'config.yaml' or as an environment variable.")
        return

    tickers_comparison = config.get('tickers', {}).get('comparison_group', [])
    focused_ticker = config.get('tickers', {}).get('focused_ticker', None)
    forecast_years = config.get('src', {}).get('forecast_years', 5)
    perpetual_growth_rate = config.get('src', {}).get('perpetual_growth_rate', 3)
    revenue_growth_rates = config.get('src', {}).get('sensitivity', {}).get('revenue_growth_rates', [0.03, 0.05, 0.07])  # 3%, 5%, 7%
    terminal_growth_rates = config.get('src', {}).get('sensitivity', {}).get('terminal_growth_rates', [0.02, 0.03, 0.04])  # 2%, 3%, 4%

    if not tickers_comparison:
        logging.error("No tickers found in the comparison group. Please update the config.yaml file.")
        print("No tickers found in the comparison group. Please update the config.yaml file.")
        return

    if not focused_ticker:
        logging.error("No focused ticker specified. Please update the config.yaml file.")
        print("No focused ticker specified. Please update the config.yaml file.")
        return

    # Step 1: Financial Ratios Comparison
    df = get_financial_data(tickers_comparison)
    print("\nFinancial Data:")
    print(df)

    # Save Financial Data to CSV
    df.to_csv('financial_data_comparison.csv', index=False)
    logging.info("Financial data saved to 'financial_data_comparison.csv'.")
    print("\nFinancial data saved to 'financial_data_comparison.csv'.")

    # Generate Financial Metrics Comparison Plots
    plot_financial_metrics_comparison(df, focused_ticker)
    plot_pe_vs_growth(df)

    # Step 2: Valuation Comparison
    valuation_df = perform_valuation(df, focused_ticker)
    print(f"\nValuation Comparison ({focused_ticker} vs Peers):")
    print(valuation_df)

    # Save Valuation Comparison to CSV
    valuation_df.to_csv('valuation_comparison.csv')
    logging.info("Valuation comparison saved to 'valuation_comparison.csv'.")
    print("\nValuation comparison saved to 'valuation_comparison.csv'.")

    # Step 3: DCF Valuation for the Focused Ticker
    print(f"\n--- DCF Valuation for {focused_ticker} ---")
    # Load assumptions from config
    assumptions = {
        'revenue_growth_rate': config.get('src', {}).get('revenue_growth_rate', 0.05),
        'ebit_margin': config.get('src', {}).get('ebit_margin', None),
        'tax_rate': config.get('src', {}).get('tax_rate', None),
        'capex_percent': config.get('src', {}).get('capex_percent', None),
        'nwc_change': config.get('src', {}).get('nwc_change', None),
    }
    dcf = DCFValuation(
        focused_ticker,
        forecast_years=forecast_years,
        perpetual_growth_rate=perpetual_growth_rate,
        fred_api_key=fred_api_key,
        config=config  # Pass config to DCFValuation
    )
    historical_data = dcf.get_historical_financials()
    if historical_data is None:
        print("No historical financial data available.")
        return

    # Run DCF Analysis with Sensitivity
    dcf_valuation, sensitivity_df, forecast_df = dcf.run_dcf_analysis(
        historical_data,
        assumptions,
        perform_sensitivity=True,
        revenue_growth_rates=revenue_growth_rates,
        terminal_growth_rates=terminal_growth_rates
    )

    # Save DCF Valuation Results to CSV
    if dcf_valuation:
        dcf_df = pd.DataFrame([dcf_valuation])
        dcf_df.to_csv('dcf_valuation.csv', index=False)
        logging.info("DCF src results saved to 'dcf_valuation.csv'.")
        print("\nDCF src results saved to 'dcf_valuation.csv'.")

    # Generate Forecast Plots
    if forecast_df is not None and not forecast_df.empty:
        plot_forecasted_fcf(forecast_df, focused_ticker)
        plot_forecast_components(forecast_df, focused_ticker)
    else:
        logging.warning("Forecast DataFrame not available for plotting.")

    # Generate Sensitivity Analysis Heatmap
    if sensitivity_df is not None and not sensitivity_df.empty:
        plot_sensitivity_analysis(sensitivity_df, focused_ticker)
    else:
        logging.warning("Sensitivity DataFrame not available for plotting.")

    # Step 4: Generate Summary Insights
    generate_summary(valuation_df, dcf_valuation, sensitivity_df)

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()
