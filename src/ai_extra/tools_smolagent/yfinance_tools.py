"""Additional tools for SmolAgents integration.

This module provides custom tools for stock data retrieval and historical data access.
"""

from datetime import date

import pandas as pd
import yfinance as yf
from smolagents import tool


@tool
def get_stock_info(symbol: str, key: str) -> dict:
    """Retrieve specific information about a stock using its ticker symbol and key.

    This tool interfaces with Yahoo Finance to fetch various stock metrics
    including price, company information, and financial indicators.
    If asked generically for 'stock price', use currentPrice.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL' for Apple)
        key: Specific metric to retrieve from the stock info.   must be one of the following: address1, city, state, zip, country, phone, website, industry, industryKey, industryDisp, sector, sectorKey, sectorDisp, longBusinessSummary, fullTimeEmployees, companyOfficers, auditRisk, boardRisk, compensationRisk, shareHolderRightsRisk, overallRisk, governanceEpochDate, compensationAsOfEpochDate, maxAge, priceHint, previousClose, open, dayLow, dayHigh, regularMarketPreviousClose, regularMarketOpen, regularMarketDayLow, regularMarketDayHigh, dividendRate, dividendYield, exDividendDate, beta, trailingPE, forwardPE, volume, regularMarketVolume, averageVolume, averageVolume10days, averageDailyVolume10Day, bid, ask, bidSize, askSize, marketCap, fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months, fiftyDayAverage, twoHundredDayAverage, currency, enterpriseValue, profitMargins, floatShares, sharesOutstanding, sharesShort, sharesShortPriorMonth, sharesShortPreviousMonthDate, dateShortInterest, sharesPercentSharesOut, heldPercentInsiders, heldPercentInstitutions, shortRatio, shortPercentOfFloat, impliedSharesOutstanding, bookValue, priceToBook, lastFiscalYearEnd, nextFiscalYearEnd, mostRecentQuarter, earningsQuarterlyGrowth, netIncomeToCommon, trailingEps, forwardEps, pegRatio, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange, SandP52WeekChange, lastDividendValue, lastDividendDate, exchange, quoteType, symbol, underlyingSymbol, shortName, longName, firstTradeDateEpochUtc, timeZoneFullName, timeZoneShortName, uuid, messageBoardId, gmtOffSetMilliseconds, currentPrice, targetHighPrice, targetLowPrice, targetMeanPrice, targetMedianPrice, recommendationMean, recommendationKey, numberOfAnalystOpinions, totalCash, totalCashPerShare, ebitda, totalDebt, quickRatio, currentRatio, totalRevenue, debtToEquity, revenuePerShare, returnOnAssets, returnOnEquity, freeCashflow, operatingCashflow, earningsGrowth, revenueGrowth, grossMargins, ebitdaMargins, operatingMargins, financialCurrency, trailingPegRatio.

    Returns:
        Dictionary containing the requested stock information
    """
    data = yf.Ticker(symbol)
    stock_info = data.info
    return stock_info[key]


@tool
def get_historical_price(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetches historical stock prices for a given symbol from 'start_date' to 'end_date'.

    Args:
        symbol (str): Stock ticker symbol.
        end_date (date): Typically today unless a specific end date is provided. End date MUST be greater than start date
        start_date (date): Set explicitly, or calculated as 'end_date - date interval' (for example, if prompted 'over the past 6 months', date interval = 6 months so start_date would be 6 months earlier than today's date). Default to '1900-01-01' if vaguely asked for historical price. Start date must always be before the current date.
    """
    try:
        data = yf.Ticker(symbol)
        hist = data.history(start=start_date, end=end_date)
        hist = hist.reset_index()
        hist[symbol] = hist["Close"]
        return hist[["Date", symbol]]
    except Exception as ex:
        from loguru import logger

        logger.error(f"failed to call get_historical_price: {ex}")
        return pd.DataFrame()
