import yfinance as yf
import pandas as pd
import numpy as np
import os
import os.path
#from datetime import datetime
import datetime
import time

# This function builds a path to the data folder where we save the data files.
def buildFilePath( tickerSymbol ):
        cwd = os.getcwd()
        pathToData = cwd +"/data/"
        fileName = pathToData + tickerSymbol + ".csv"
        return fileName

# Download a list of 500 tickers from wikipedia:
def list_wikipedia_sp500() -> pd.DataFrame:
        # Ref: https://stackoverflow.com/a/75845569/
        url = 'https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp500ListWiki = pd.read_html(url, attrs={'id': 'constituents'}, index_col='Symbol')[0]
        sp500ListWiki.reset_index(inplace=True)
        fileName = buildFilePath( "sp500ListWiki" )
        sp500ListWiki.to_csv(fileName, header=True, index=False)
        return sp500ListWiki 

#================================================
# Download a list of 500 tickers from wikipedia:
# Run this code only once:
#=================================================
#sp500WikiList = list_wikipedia_sp500()
#print(sp500WikiList)

# To use a custom requests session, pass a session= argument to the Ticker constructor. 
# This allows for caching calls to the API as well as a custom way to modify requests via the User-agent header.

# Combine requests_cache with rate-limiting to avoid triggering Yahoo’s rate-limiter/blocker that can corrupt data.
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
   pass

session = CachedLimiterSession(
   limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
   bucket_class=MemoryQueueBucket,
   backend=SQLiteCache("yfinance.cache"),
)

notAvailableYf = []
def getTickerYf( tickerSymbol ):
        # Create a Ticker object
        print("\nLine 53: \nTicker:", tickerSymbol, "connect to YF.")
        ticker = yf.Ticker(tickerSymbol, session = session )
        #print("\nLine 57: \nticker object:", type(ticker), ticker.info)
        # The scraped response will be stored in the cache
        ticker.actions
        # Check if the .csv file exists. If it does, update it. Otherwise period=max.
        fileName = buildFilePath( tickerSymbol )
        if os.path.isfile(fileName): # if true, file exits. It means I downloaded before and can update now.
                print("\nLine 60: \nTicker:", tickerSymbol, "file exists. Download update.")
                # Get the start date from the last row of the existing file.
                tempSymbolDF = pd.read_csv(fileName)
                nRowTempDF = tempSymbolDF.shape[0] # number of rows in the DF.
                # https://www.geeksforgeeks.org/python-convert-string-to-datetime-and-vice-versa/
                #format = '%Y-%m-%d'
                #startDate = datetime.datetime.strptime(tempSymbolDF.Date[nRowTempDF-1], format).date()
                startDate = tempSymbolDF.Date[nRowTempDF-1]
                #startDate = datetime.datetime(2024,12,12) # This is the last date on the existing file
                #endDate = datetime.datetime.strftime(datetime.datetime.now() - datetime.timedelta(1), "%Y-%m-%d") # date time as string
                endDate = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d") # date time as string
                histoData = ticker.history(start=startDate, end=endDate).iloc[1:,0:6]
                print("\nLine 72: \nstartDate:", startDate, "endDate=", endDate, histoData.tail())
                
        else: # the file does not exist. Download max range.
                print("\nLine 75: \nTicker:", tickerSymbol, "file does not exist. Download max.")
                histoData = ticker.history(period="max")  # data for the last year
        # Fetch historical market data:
        
        if( histoData.shape[0] == 0): # no rows, no data.
                print("\nLine 68: \nTicker:", tickerSymbol, "not available.")
                notAvailableYf.append(tickerSymbol) 
        else:         
                # Reset index and treat Date column as a date column.
                histoData.reset_index(inplace=True)
                histoData["Date"] = histoData["Date"].dt.strftime('%Y-%m-%d')
                                
                if os.path.isfile(fileName): #If file exists, add the update to the end of DF.
                        tempSymbolDF = tempSymbolDF._append(histoData.iloc[:histoData.shape[0]+2, ], ignore_index= True)
                        tempSymbolDF.to_csv(fileName,header=True, index=False)
                        print("\nLine 85: \nTicker:", tickerSymbol, "\n Save updated DF.")
                else:# file does not exist.
                        print("\nLine 87: \nTicker:", tickerSymbol, "\n Save max period.")
                        histoData.iloc[:histoData.shape[0], ].to_csv(fileName, header=True, index=False)
        return        
        
# Add in the for loop:
#time.sleep(seconds) 

#============================================
# Download Indices: SP500, Nasdaq100, DJIA time series only once:
marketIndices = ["^GSPC", "^DJI", "^IXIC", "^GDAXI", "^FTSE", "^FCHI", "^N225", "^HSI", "^STI"] 
bondIndices = ["^TYX", "^TNX","^FVX","^IRX"]
cmdtyIndices= ["GLD","CL=F","DX-Y.NYB"]
#==============================================
def getTickerWraperYf( tickerList ):
        for ticker in tickerList:
                getTickerYf(ticker)
        return
# Download once:
#getTickerWraperYf(marketIndices)
#getTickerWraperYf(cmdtyIndices)


#===============================================
# Load the list of 500 tickers, just once.
sp500Stocks = list(pd.read_csv(buildFilePath("sp500ListWiki"))["Symbol"])
nTickers = len(sp500Stocks)
#print("\nLine 84: \ntickerList:\n", tickerList, "\nTickers=", nTickers )

# Main loop: Retrieve data from yf one time.
t0 = time.time()
# yahoo finance syntax: ["BF-B", "BRK-B"]
# wikipedia list syntax: ["BF.B", "BRK.B"]
# Dictionary to map from wikipedia to yf. This is the same syntax used to save the files:
syntaxDict = { "BF.B":"BF-B", "BRK.B":"BRK-B"}
yfSyntaxRejected = ["BF-B", "BRK-B"] #BF.B, BRK.B
def getAllTickers( tickerList ):
        for i in range( len(tickerList)):
                tickerSymbolTemp =  tickerList[i]
                if (tickerSymbolTemp == "BF.B"):
                        tickerSymbolTemp = "BF-B"
                        continue
                if (tickerSymbolTemp == "BRK.B"):
                        tickerSymbolTemp = "BRK-B"        , 
                        continue
                #tickerSymbolTemp =  testSample[i]
                getTickerYf( tickerSymbolTemp )
t1 = time.time()
#getAllTickers( sp500Stocks )
#getAllTickers(yfSyntaxRejected)
print("\nLine 141: total time=", (t1-t0) , "\nnotAvailableYf=", notAvailableYf )
#===========================================================================================

# Test one stock.
#===================
#tickerSymbol = "LEN"
#getTickerYf(tickerSymbol)
#print("\nLine 145: tickerSymbol=", tickerSymbol )

# Clean the data first:
# 1) Take Sp500 GSPC as a reference. Clean GSPC first.
# Volume begins to be reported on 1950-01-03. Before then it is 0. 
# At that same time open, high, low, close are properly reported. Before then, they are all the same.
# I will save the original download and start the series on 1950-01-03.
def newColNames( oldNamesList, ticker ):
        newColNamesL = ["Date"]
        for j in range(1,len(oldNamesList)):
                deleteChar = "^.-="
                for char in deleteChar:
                        ticker = ticker.replace(char, "")
                newName = ticker+oldNamesList[j]
                newColNamesL.append(newName)
        return newColNamesL



# If the ticker was processed before, it wont be processed again.
# Delete it first.
naV = ["na", "NA","NaN", 0, "0"]
def createSP500Clean():
        tickerSymbol = "^GSPC"
        if( not os.path.isfile( buildFilePath(tickerSymbol+"1") )):
                fileName = buildFilePath( tickerSymbol )
                mainDF = pd.read_csv( fileName )
                # Save original download as ^GSPC
                mainDFLen = mainDF.shape[0]
                startRow = np.where(mainDF.Date == "1950-01-03")

                mainDF = mainDF.iloc[startRow[0][0]:, 0:6]
                mainDFLen = mainDF.shape[0]
                # Fix the 0 at open in 1962:
                
                for i in range(mainDFLen):
                        if( mainDF.iloc[i, 1] in naV ):
                                mainDF.iloc[i, 1] = mainDF.iloc[ (i-1), 4]
                                #print("\nLine 167: i=", i)
                # Add ticker symbol to the name of the columns: 
                mainDF.columns = newColNames(mainDF.columns, tickerSymbol)

                mainDF.to_csv( buildFilePath(tickerSymbol+"1"), header=True, index=False)

                print("\nLine 168: startRow=", startRow[0][0], "\nmaindDF=\n", mainDF)
                # Now with SP500 clean, the date is the benchmark for all other stocks:
                #sp500Stocks = list(pd.read_csv(buildFilePath("sp500ListWiki"))["Symbol"])
                #nTickers = len(sp500Stocks)
        else: 
                fileName = buildFilePath(tickerSymbol+"1")
                mainDF = pd.read_csv( fileName )
        return mainDF

mainDF = createSP500Clean()
mainDate = list(mainDF.Date)
mainDate0 = mainDate[0]
mainDFLen = mainDF.shape[0]

def cleanSecurityData(securityList):
        securityListLen = len(securityList)
        for i in range(securityListLen): # nTickers
                tickerSymbol = securityList[i]
                if (tickerSymbol == "BF.B"):
                        tickerSymbol = "BF-B"
                if (tickerSymbol == "BRK.B"):
                        tickerSymbol = "BRK-B"
                if (tickerSymbol in ["CTLT", "ETR", "LEN"]):
                        continue
                # When the security has been processed the file is saved with a 1 next to the symbol.
                criteria1 = not os.path.isfile( buildFilePath(tickerSymbol+"1") )
                # The security is not ^GSPC.
                criteria2 = (tickerSymbol != "^GSPC")
                if(criteria1 and criteria2 ):
                        #tickerSymbol = securityList[i]
                        fileName = buildFilePath( tickerSymbol )
                        securityDF = pd.read_csv( fileName )
                        # Check the earliest date in the security:
                        startDate = securityDF.Date[0]
                        #securityDFLen = securityDF.shape[0]
                        # This is the row in SP500 to start matching:
                        targetRow = np.where(mainDF.Date == startDate)[0][0]
                        print("\nLine 218: targetRow=", targetRow, "\ntickerSymbol=", tickerSymbol)
                        # Match every day of trading in the SP500 to the security:
                        tempDate = list(mainDF.iloc[targetRow:, 0].copy())
                        tempOpen=[]
                        tempHigh=[]
                        tempLow=[]
                        tempClose=[]
                        tempVolume=[]
                        for j in range(targetRow, mainDFLen):
                                #print("\nLine 228: securityDF date=", securityDF.iloc[ j-targetRow , 0], 
                                #      "\ntempdatelist=", tempDate[j-targetRow])
                                nextIndexV = np.where(securityDF.Date == tempDate[j-targetRow])[0]
                                if( len(nextIndexV)==0 ): # I did not find the date.
                                        #print("\nLine 228: no match at j=", j, "\ndate= ", tempDate[j-targetRow])
                                        tempOpen.append( tempOpen[ j-targetRow-1] )
                                        tempHigh.append(tempHigh[ j-targetRow-1] )
                                        tempLow.append(tempLow[ j-targetRow-1] )
                                        tempClose.append(tempClose[ j-targetRow-1] )
                                        tempVolume.append(tempVolume[ j-targetRow-1] )
                                else: # The row matches the date in the SP500.
                                        
                                        # If close is na, assign the close of the previous day.
                                        if( securityDF.iloc[ nextIndexV[0] , 4] in naV):
                                                tempClose.append(tempClose[-1])
                                        else:
                                                tempClose.append(securityDF.iloc[nextIndexV[0], 4])
                                        
                                        # If High in NA, assign the close of that day:
                                        if( securityDF.iloc[ nextIndexV[0], 2] in naV ):
                                                tempHigh.append(tempClose[-1])
                                        else:
                                                tempHigh.append(securityDF.iloc[nextIndexV[0], 2])

                                        # If Low in NA, assign the close of that day:
                                        if(securityDF.iloc[nextIndexV[0], 3] in naV):
                                                tempLow.append(tempClose[-1])
                                        else: 
                                                tempLow.append(securityDF.iloc[nextIndexV[0], 3])
                                        
                                        # If open is na, assign the low of that day.
                                        if( securityDF.iloc[nextIndexV[0], 1] in naV):
                                                tempOpen.append(tempLow[-1] )
                                        else:         
                                                tempOpen.append(securityDF.iloc[nextIndexV[0], 1] )

                                        # If Volume in NA, assign the Volume of the previous day:
                                        if( securityDF.iloc[nextIndexV[0], 5] in naV and j != targetRow):
                                                tempVolume.append(tempVolume[-1])
                                        else:
                                                tempVolume.append(securityDF.iloc[nextIndexV[0], 5])
                                        
                        tempDF = pd.DataFrame( list(zip(tempDate, tempOpen, tempHigh, tempLow, tempClose, tempVolume)), 
                                              columns= ["Date", "Open", "High", "Low", "Close", "Volume"])
                        # Count the number of zeros in Volume, and if more than 100, drop it altogether.
                        cTempVol = tempVolume.count(0) 
                        if( cTempVol >= 100 ):
                                tempDF = tempDF.drop("Volume", axis=1)
                                print("\nLine 288: Volume dropped on tickerSymbol=", tickerSymbol)        
                        tempDF.columns = newColNames(tempDF.columns, tickerSymbol)
                        tempDF.to_csv(buildFilePath(tickerSymbol+"1"), header=True, index=False)
                        print("\nLine 259: tickerSymbol=", tickerSymbol, "saved, completed.")
                else: # I already processed this security. I have a file on it.
                        print("\nLine 243: ticker=", securityList[i], "already has a processed file")
        return
# Flag the security complete so that we dont come back to it.

# test for 1:
#cleanSecurityData(["^TYX"])

cleanSecurityData(cmdtyIndices)
cleanSecurityData(marketIndices)
cleanSecurityData(bondIndices)
cleanSecurityData(sp500Stocks)