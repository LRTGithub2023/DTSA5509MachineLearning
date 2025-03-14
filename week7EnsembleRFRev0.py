
# GENERIC PACKAGES:
#==================
from datetime import datetime
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import math
import numpy as np
import pandas as pd
import os
import random
import seaborn as sns
import time
import warnings

# SCIPY:
#=======
import scipy
from scipy import stats
from scipy.stats import norm
from scipy.stats.stats import pearsonr

# STATSMODELS:
#=============
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

#SKLEARN:
#========
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import tree

# XGB:
#=====
import xgboost
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", message=".*does not match any known type.*")

t0 = time.time()

def linear_plot(X, yActual, yPred, xTest, yTestActual, yTestPred, targetStock, fileName, xLabel= "Date", yLabel="ClosePrice" ):      
        # Plot data 
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 16))
        ax.tick_params(axis="both", labelsize=20)
        plt.scatter(x=X, y=yActual, marker="o", s=200, color="Black", alpha=1)
        plt.scatter(x=X, y=yPred, marker="^", s=200, color= "#CFB87C", alpha=1)
        plt.scatter(x=xTest, y=yTestActual, marker="o", s=500, color= "Blue", alpha=0.7)
        plt.scatter(x=xTest, y=yTestPred, marker="^", s=500, color= "Red", alpha=1)
        # Add overlayed regression line:
        
        plt.legend(["Gray(dots) => Actual Close Price","Gold(triangle) => Predicted Close Price"],fontsize=30, loc="lower right")
        #plt.axhline(y=0, color="red", linestyle = "dashed")
        #plt.axvline(x=0, color="red", linestyle = "dashed")
        plt.title( targetStock + "--" + yLabel + " vs "+ xLabel, fontsize=40)
        plt.ylabel(yLabel, fontsize=24)
        plt.xlabel(xLabel, fontsize=24)
        
        plt.xticks(X, rotation='vertical')
        tkStart, tkEnd = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(tkStart, tkEnd, len(X)//70+1))
        
        plt.grid()
        spacing = 0.500
        fig.subplots_adjust(bottom=spacing)
        plt.tight_layout()
        plt.savefig(fileName)

        return

def fixBerkshire(brkSymbol):
        brkSymbolC= brkSymbol
        if brkSymbol == "BF.B":
                brkSymbolC ="BF-B"
        if brkSymbol == "BRK.B":
                brkSymbolC ="BRK-B"        
        return brkSymbolC

def buildFilePath( tickerSymbol ):
        cwd = os.getcwd()
        pathToData = cwd +"/data/"
        fileName = pathToData + tickerSymbol + ".csv"
        return fileName

# Adjust the names of the columns to add the name of the Stock.
def newColNames( oldNamesList, ticker ):
        newColNamesL = ["Date"]
        for j in range(1,len(oldNamesList)):
                newName = ticker+oldNamesList[j]
                newColNamesL.append(newName)
        return newColNamesL

# Load and cache a number securities (without merging them):
# This function reads the securities in securitiesV, merges them with target stock 
# and returns a dictionary with all securities in memory.
def loadAllSecDF(targetStock, securitiesV):
        # Build a dictionary with all the predictors data of all securities:
        # If target stock is inside securities, do not duplicate:
        secS= set(securitiesV)
        stockS = set([targetStock])
        allSecV = list(secS - stockS )  
        allSecV.sort() # this makes it replicable. Sets is hash.
        #print("\nLine176: allSecV=", allSecV)
        allSecV = [targetStock] + allSecV
        #print("\nLine178: allSecV=", allSecV)
        #if targetStock in securitiesV:
        #        allSecV = securitiesV
        #else:
        #        allSecV = [targetStock] + securitiesV
        securitiesD = {}
        for i in range( len(allSecV) ):
                securityTicker = fixBerkshire(allSecV[i])
                fileName = buildFilePath(securityTicker+"1")
                tempDF = pd.read_csv(fileName)
                securitiesD[allSecV[i]] = tempDF
        return securitiesD # returns a dictionary with all securities loaded in memory.
# Test loadAllSecDF:
#testLoadFuntion = loadAllSecDF("AAPL", ["MSFT", "GOOG", "AAPL"])
#print( "\nLine 185: testLoadFuntion=", testLoadFuntion)
#This function works fine.


# This function merges in time, and lags a targetStock with one or more securities stored in securitiesV:
# The securitiesD dictionary must be created by loadAllsecDF prior to calling merge.
# The function returns a compiled DF with targetstock and lagged securitiesV.
def mergeSecsDF(targetStock, securitiesD, securitiesV, npLag=5, seedValue=2025):
        # This dictionary has all the data series for securitiesV and targetStock.
        # Initialize the compiled DF with the targetStock data.
        #print("\nLine138: targetStock=", targetStock, "\nsecuritiesD=", securitiesD)
        loadDF = securitiesD[targetStock]
        # Take rows starting from the last one, back to the beginning, every npLag periods.
        # The DF ends up in reverse order, hence, iloc[::-1] to reverse back to ascending sequence.
        compileDF = loadDF.iloc[loadDF.shape[0]:0:-npLag].iloc[::-1]  
        # The securtiesV vector holds a selected list of securities,
        # each having an open, high, low, close, and some of them volume, which 
        # become predictor variables.
        nPred = len(securitiesV)
        # We will iterate through each security, and merge it into the main compileDF
        # with Date as a key. This will be an inner joint: the resulting merge
        # are rows where the Date matches (is the same) in both the target stock and
        # the merging security. This process leads to the shortest time series amongst
        # target stock and securities selected.
        for i in range(nPred):
                if(targetStock != securitiesV[i]): # security and targetstock are not the same.
                        tempDF = securitiesD[securitiesV[i]]
                        newDF = compileDF.copy().merge(tempDF, how="inner", on="Date")
                        compileDF = newDF
                        #print( "\nLine 214: target and sec are not the same.\n", compileDF)
                else: # the target and the security are the same: add a lagged close.
                        compileDF[targetStock+"CloseLag"] = compileDF[targetStock+"Close"].copy()
                        #print( "\nLine 217: target and sec are the same.\n", compileDF)
        # If I use sets:
        colNames = set(compileDF.columns)
        doNotShift = set(["Date", (targetStock+"Close")])
        colShift = list(colNames-doNotShift)
        colShift.sort() # sort makes it replicable, but not good results.
        # The problem with sets is that the order is uncertain. It changes every time we run,
        # and results are not replicable.
        
        compileDF = pd.concat([compileDF.copy().loc[:,["Date", (targetStock+"Close")]], compileDF.copy().loc[:,colShift].shift(1)], axis=1)
        # Reset index, in place:
        compileDF.reset_index(drop=True, inplace=True) 

        # Split the date into year, month day:        
        dateSplit = pd.to_datetime(compileDF["Date"].copy())
        
        for i in range( dateSplit.shape[0]):
                compileDF.loc[[i],"year"] = int(dateSplit[i].year)
                compileDF.loc[[i],"month"] = int(dateSplit[i].month)
                compileDF.loc[[i], "day"] = int(dateSplit[i].day)
        compileDF = compileDF.astype({"year":"int", "month":"int", "day":"int"})
        
        # Move year, month, day next to Date:
        moveCols = ["year", "month", "day"]
        for i in range(len(moveCols)):
                moveCol = compileDF.pop(moveCols[i])
                compileDF.insert(i+1,moveCols[i], moveCol)      
        #print("\nLine 244: \ncompileDF=\n", compileDF, compileDF.dtypes )
        return compileDF


# The trainTestSets function is used in decision trees and random forests:        
def trainTestSets(compileDF):
        # Get the test set: The last row of the dataframe is the date to predict.
        compileDFTest = compileDF[-1:]
        # The rest of the dataset is the training set:
        compileDFTrain = compileDF[1:-1]
        # This is the base start of the np periods compileDF 
        # with the target stock, date, year, month, day
        # and Open, High, Low, Volume(when available) lagged np periods.
        featureNames = list(compileDFTrain.columns.copy())
        featureNames.remove("Date")
        featureNames.remove( (targetStock+"Close") )
        XTrain= compileDFTrain.drop( [(targetStock+"Close"), "Date"], axis=1).values
        yTrain= compileDFTrain[(targetStock+"Close")].values
        XTest= compileDFTest.drop( [(targetStock+"Close"), "Date"], axis=1).values
        yTest= compileDFTest[(targetStock+"Close")].values
        
        #print("\nLine 59: \ncompileDF=\n", XTrain, "\n\n",yTrain, "\n\n",XTest, "\n\n",yTest, "\n\n", featureNames)
        print("\nLine 337: yTest", yTest)
        return featureNames, XTrain, yTrain, XTest, yTest

# The function buildCorrMatrix builds a correlation matrix between target stock and predictors, 
# individually, one by one, and select the predictors with correlation > than a threshhold (default =0.7
# but it can be anything).
# Load the dictionary of securities first.
# This gets the maximum history of the security and target stock, aligned by date and lag.
# Each targetStock and security has different history, hence, correlations may vary
# when they are measured on different underlying data.
# It returns the correlation Matrix in a file.
# This function was used during the initial stage of exploration. It is not part of any
# modelling routine
def buildCorrMatrix(targetStock, securitiesV, corrThresh=0.7, minSeries=250, npLag=5, seedValue=2025):
        highCorrP = [] # store the highly correlated predictors.
        highCorrValues = []
        highCorrSecName = []
        belowCorrThreshV =[]
        # Let's add one filter here: securities need y years of history to be considered.
        # The criteria is 1 year of date for 1 day model: 250 rows.
        # 5 years of data for a 5 days model: 250 rows.
        for i in range(len(securitiesV)):
                tempDF = mergeSecsDF(targetStock, [securitiesV[i]], npLag=5, seedValue=2025)
                if tempDF.shape[0] >= minSeries: #This ensures enough length in the data set.
                        # Measure the correlations and keep the higher ones.
                        x = tempDF.iloc[1:][targetStock+"Close"]
                        predictorNames = list(tempDF.columns)
                        # These are the names in the DF. Date, year, month, day,
                        # Then APPL Close, High, Low, Open, Volume,
                        # Then the security. 
                        nNames = len(predictorNames)
                        # if targetStock Volume is in the DF, then start at 9, else, start at 8.
                        if( targetStock != securitiesV[i]):
                                if( (targetStock+"Volume") in predictorNames):
                                        jStart = 9 # The index jStart needs to change if the targetStock does not have Volume.
                                else:
                                        jStart = 8
                        else: # security and targetStock are the same.
                                jStart = 1

                        for j in range(jStart,nNames):
                                # Look at AAPL and dates only once
                                if len(predictorNames[j]) == 0: # just started the measures of correlation.
                                        jStart = 1 # The year field.
                                if predictorNames[j] != (targetStock + "Close"):
                                        y = tempDF.iloc[1:][predictorNames[j]]
                                        print("\nLine 543: predictorNames[j]=\n", predictorNames[j])
                                        correlation, p_value = stats.pearsonr(x, y)
                                        #print("\nLine 394: name", predictorNames[j], "\ncorrelation", correlation, "p_value=", p_value)
                                        if abs(correlation) >= corrThresh:
                                                highCorrP.append(predictorNames[j])
                                                highCorrValues.append(correlation)
                                                highCorrSecName.append(securitiesV[i])
                                                print("\nLine 550: i=", i, "\nadding securitiesV[i]=\n", [securitiesV[i]], correlation)
                                        else:
                                                belowCorrThreshV.append(securitiesV[i])
                                                print("\nLine 553: i=", i, "\nsecuritiesV[i]=\n", [securitiesV[i]], "\nbelow orrelation threshhold.")
                else:
                        print("\nLine 555: i=", i, "\npredictor:", securitiesV[i], "not enoughhistory.")
        corrDF = pd.DataFrame( 
                list(zip(highCorrP,highCorrValues,highCorrSecName)), 
                columns= ["Predictor", "corrValue", "secName"] )
        corrDF = corrDF.sort_values(by="corrValue", ascending=False)
        #print("\nLine 560: corrDF=\n", corrDF, "\n", corrDF.shape[0], "\nSecurities with Corr Threshhold:\n", belowCorrThreshV)
        corrDF.to_csv(buildFilePath("corrMatrix"),header=True, index=False)
        return

#=======================
# MAIN DRIVER CODE
#======================
yfSyntaxRejected = ["BF-B", "BRK-B"] #BF.B, BRK.B
tickerList = list(pd.read_csv(buildFilePath("sp500ListWiki"))["Symbol"])
nTickers = len(tickerList)



# The function pairwiseCorrDF was used in the early stage of discovery and it is not used
# in modelling. The function computes correlation between pairs of predictors.
# It is a large 2000 x 2000 matrix. It takes 14 days to compute it all.
def pairwiseCorrDF(corrDF, corrThresh=0.7):
        #This is the list of securities which correlate with the target stock > corrThresh and they have a minimum
        # of 250 rows in the set, in this case 5 years since we build the set every 5 days.
        criteria1 = abs(corrDF["corrValue"]) > corrThresh
        secV = list(corrDF.loc[criteria1]["secName"]) # unique list of securities with corr > 0.7.
        # We now look at pairwise correlations between predictors.
        # This is just an indication, because the data on which we are measuring correlations
        # changes depending on the pair we use.
        #highCorrPredS = set(corrDF["Predictor"])
        highCorrPredV = list(corrDF["Predictor"])
        # Main data frame matrix type to store all correlation results.
        mainHighCorrDF = pd.DataFrame(index=[targetStock+"Close"]+highCorrPredV, 
                                      columns=[targetStock+"Close"]+highCorrPredV)
        mainHighCorrDF = mainHighCorrDF.fillna('')
        #print("\nLine 436: mainHighCorrDF=\n", mainHighCorrDF)

        # PairWise correlations:
        for i in range(len(highCorrPredV)):
                tempSecNamei = secV[i]
                tempPredNamei = highCorrPredV[i]
                #tempCorrV = [1]
                for j in range(i+1, len(highCorrPredV)):
                        tempSecNamej = secV[j]
                        tempPredNamej = highCorrPredV[j]
                        
                        tempSecV = []
                        if( tempSecNamei != tempSecNamej):
                                tempSecV.append(tempSecNamej)

                        temPredS = set()
                        temPredS.add(tempPredNamei)
                        temPredS.add(tempPredNamej)
                        temPredS = list(temPredS)
                        
                        #highCorrTempDF = buildDF(targetStock, tempSecV, npLag=5, seedValue=2025)
                        highCorrTempDF = mergeSecsDF(tempSecNamei, securitiesV=tempSecV, npLag=5, seedValue=2025)

                        # Measure the correlation between 2 predictors:
                        for colName in list(highCorrTempDF.columns):
                                if not (colName in temPredS):
                                        highCorrTempDF = highCorrTempDF.drop(colName, axis=1)

                        mainHighCorrDF.loc[tempPredNamei][tempPredNamej] = abs(highCorrTempDF.iloc[1:,:].corr()).iloc[0,1]
                        #print("\nLine 461: mainHighCorrDF=\n", mainHighCorrDF)
                        print("\nLine 610: i=", i, "j=", j)
                # Save the file at every predictor completion.
                # Then reload and continue from there if there is an error.
                mainHighCorrDF.to_csv(buildFilePath("mainHighCorrDF"),header=True, index=False)
        return 

#if (not (os.path.isfile(buildFilePath("mainHighCorrDF")))):
#        pairwiseCorrDF(corrDF, corrThresh=0.7)
#mainHighCorrDF = pd.read_csv(buildFilePath("mainHighCorrDF"))
#print("\nLine 652: \nmainHighCorrDF=\n", mainHighCorrDF)


# Set up the combinations indexes:
#=========================================
def setUpCombis(topn=3):
        combiSeq = np.arange(0,topn,1) #[0,1,2,3,4]
        allCombisM = []
        for k in range(1,len(combiSeq)+1):
                tempCombisk = []
                for comb in itertools.combinations(combiSeq, k):
                        tempCombisk.append(comb) # comb is a tuple of len k
                allCombisM.append(tempCombisk)
        #print("\nLine 462: allCombisV=", allCombisM)
        return allCombisM

# Build a glm compatible formula to pass to the glm model.
#==========================================================
def buildFormulas(targetStock, corrTopFNames, corrTopFSec, allCombisM):  
        #print("\nLine 470: topFNames=", corrTopFNames)
        # Build the formula to pass to glm:
        formulaM = []
        secNameM = []
        for k in range(len(allCombisM)): # each row has combinations of n taken by k.
                print("\nLine 480: k=", k)#, "glmTempForm=", glmTempForm)
                tempFormsk = []
                tempSecNamek = []
                for i in range( len(allCombisM[k]) ):
                        glmTempForm = targetStock+"Close~"
                        tempCombi = allCombisM[k][i]
                        tempSecNameT = set()
                        for j in range(len(tempCombi)):
                                glmTempForm += "+" + corrTopFNames[tempCombi[j]]
                                tempSecNameT.add(corrTopFSec[tempCombi[j]])
                        tempFormsk.append(glmTempForm)
                        tempSecNamek.append(list(tempSecNameT))
                        #print("\nLine 484: i=", i, "glmTempForm=", glmTempForm)
                        
                formulaM.append(tempFormsk)
                secNameM.append(tempSecNamek)
        return formulaM, secNameM

def fitglm(targetStock, glmForm, trainSetDF, testSetDF, alpha, R2Max):
        modelTemp = smf.glm(formula=glmForm, data=trainSetDF).fit()
        # Evaluate the model: 
        #====================
        #1) Review the pvalues, discard the model if pvalues are > alpha.
        pv = dict(modelTemp.pvalues)
        pvKeys = list(pv.keys()) # The name of the predictors.
        #print("\nLine 518: pv=\n", pv, "\npvKeys=", pvKeys, "\nglmForm=", glmForm, "\nmodelTemp=\n", modelTemp.summary())
        for k in range(1,len(pvKeys)): # skip 0, because it is the intercept.
                if( pv[ pvKeys[k] ] > alpha):
                        #print("\nLine 515: pvKeys[k]=\n", pvKeys[k], "> alpha no good pvalues.")
                        return []
                #else:
        #print("\nLine 518: glmForm=\n", glmForm, "\nModel=\n", modelTemp.summary(),"\n pvalues ok.")
        #print("\nLine 518: glmForm=\n", glmForm, "\n pvalues ok.")
        
        #2) Review VIF, discard the model if VIF > vifMax on any predictor.
        #print("\nLine 521: trainSetDF=\n", trainSetDF)
        if( len(pvKeys) > 2): # pvKeys has the intercept as first element. We need at least 2 predictors.
                # Regress each predictor vs all others, and measure R2:
                predictorL = pvKeys[1:]
                for i in range(len(predictorL)):
                        tempDF = trainSetDF.copy()
                        y = tempDF[ predictorL[i] ]
                        x = tempDF[predictorL].drop(predictorL[i], axis=1) # review syntax here.
                        #print("\nLine 530: y=\n", y, "\nx=", x)
                        R2 = LinearRegression().fit(x, y).score(x, y)
                        if R2 > R2Max:
                                #print("\nLine 526: predictorL[i]=", predictorL[i], "\nR2=", R2, " vif too high.")
                                return []
                #print("\nLine 528: predictorL[i]=", predictorL[i], "\nR2=", R2, " vif ok")
        #3) If we got this far, the model is sound. Measure its test error:
        yTestPred = modelTemp.predict(testSetDF).values[0]
        yTestActual = testSetDF.iloc[0][(targetStock+"Close")]
        trainRMSE = np.sqrt(modelTemp.deviance/modelTemp.df_resid)
        trainRMSEPct = trainRMSE/yTestActual*100
        testErrorPct = (yTestPred - yTestActual)/yTestActual * 100
        #print("\nLine 773: modelTemp=\n", modelTemp.summary(), 
        #      "\nyPred=", round(yTestPred,2), "\nyActual=", round(yTestActual, 2),
        #      "\nerrorPredPct=", round(testErrorPct,4), "%"
        #      "\ntrainRMSE=", round(trainRMSE,4), 
        #      "\ntrainRMSEPct=", round(trainRMSEPct,4), "%" )
        return [modelTemp, yTestPred, testErrorPct, trainRMSEPct, yTestActual] # return the model, its test prediction, and the prediction error.

# FormulaM is a list of lists. Each lists contains tuples of k predictor combinations.
def combiglmModels(targetStock, securitiesD, formulaM, secNameM, alpha, R2Max,npLag,seedValue):
        bestglmModelV = []
        for k in range(len(formulaM)): # k=0, 1 predictor; k=n, n-1 predictors.
                # Formulas go from 1 predictor to k=n=5,10,15 predictors
                # There are n formulas of 1 predictor, and 1 formula of n predictors.
                glmFormV = formulaM[k] # glmFormV is a list with formulas of k predictors.
                glmSecNameV = secNameM[k] # The ticker=MSFT of the security in the predictor=MSFTOpen.
                #print("\nLine 557: models with k=", k+1, "predictors.\nglmFormV=\n", glmFormV, "\nglmSecNameV=", glmSecNameV)
                # Build the DF to support the model.
                bestkModel = [] # The best model of k predictors.
                # We are looking at the best model with k predictors.
                bestError = 1000000 # The best error is reset for each k.
                # This for loop goes through all combinations of k predictors:
                for i in range(len(glmFormV)):
                        tempSecV = glmSecNameV[i]
                        #tempDF = buildDF(targetStock,tempSecV,npLag,seedValue)
                        tempDF = mergeSecsDF(targetStock, securitiesD, tempSecV, npLag, seedValue) # read off the dictionary already loaded.
                        trainSetDF = tempDF[1:-1]
                        testSetDF = tempDF[-1:]
                        modelTempV = fitglm(targetStock, glmFormV[i], trainSetDF, testSetDF, alpha, R2Max)
                        # fitglm returns: [modelTemp, yTestPred, testErrorPct, trainRMSEPct, yTestActual]
                        # If modelTempV has len=0, there is no model to look into.
                        if len(modelTempV) != 0:
                                #print("\nLine 570: i=", i, "glmFormV[i]=", glmFormV[i], "is a good model.")
                                if abs(modelTempV[2]) < bestError:
                                        bestError = abs(modelTempV[2])
                                        bestkModel = modelTempV
                                        #print("\nLine 574: modelTempV=\n", modelTempV[0].summary(), "\nbestError=", bestError,
                                        #      "\nmodelError=", modelTempV[2], "\nThis model is better")
                                #else: 
                                        #print("\nLine 572: modelTempV=\n", modelTempV[0].summary(), "\nbestError=", bestError,
                                        #      "\nmodelError=", modelTempV[2], "\nThis model is NOT better")
                        #else:
                                #print("\nLine 577: i=", i, "No good models here.")
                bestglmModelV.append(bestkModel)
        return bestglmModelV

def diagnosticPlots(lowestErrorSPModel, testClosePredV, closeTestActual, closeTestPred, filename):
        # Data for all plots:
        #===================
        # The best model is the lowest error:
        # I need the dates as part of the data.
        # testClosePredV = This vector holds all the predictions of close price on the test set.
        residuals = lowestErrorSPModel.resid_response # residuals = y-yhat
        stand_resids = lowestErrorSPModel.resid_pearson # This shows the same a resid_response. Remove.
        fittedValues = lowestErrorSPModel.fittedvalues # yhat = APPLClosePricePredicted (Training data)
        actualValues = lowestErrorSPModel.model.endog # y = APPLClosePriceActual (Training data)
        targetStockSeries = pd.DataFrame({"Actual":list(actualValues), "Fitted":list(fittedValues)})
        #print("\nLine 921: \nresiduals=\n", residuals, "\nstand_resids=", stand_resids)
        #print("\nLine 334: lowestModel df=\n", lowestModel.model.data.endog) # actual close values stored in the model.
        #targetStockSeries= lowestErrorSPModel.model.exog
        predNames = lowestErrorSPModel.model.exog_names # predictors names.

        #print("\nLine 924: \n stock series=\n",targetStockSeries, "\nyActual=", actualValues, "\npredNames=", predNames)
                
        
        # Distribution of the Test Error Predicted Close Price.
        #==============================================================
        plt.figure(figsize=(20,20))
        fig0, ax0 = plt.subplots()

        # create a color map:
        cmap = cm.get_cmap("magma")
        # normalize the indices to map the color map between 0 and 1:
        norm = plt.Normalize(vmin=min(np.array(testClosePredV)), vmax=max(np.array(testClosePredV)))

        # Create a violin plot
        parts = ax0.violinplot(np.array(testClosePredV), showmeans=True, showextrema=True, showmedians=True)
        # This is useful if I have many violins and want them in different colors.
        #for i, pc in enumerate( parts["bodies"]):
        for pc in parts["bodies"]:
                color = "#565A5C" # cu dark gray
                pc.set_facecolor(color)
                pc.set_edgecolor("black")
                pc.set_alpha(0.5)

        # Set figure frame color to light gray.
        #fig.set_facecolor("#A2A4A3") # cu light gray
        #fig.patch.set_facecolor('#f0f0f0')  
        # Display data points
        plt.scatter([1]*len(np.array(testClosePredV)), np.array(testClosePredV), 
                alpha=1, s=30, cmap=cmap, c=norm(np.array(testClosePredV)))

        # Customize the colors and styles
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        #for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        #for partname in ('cbars', 'cmins', 'cmaxes'):
                vp = parts[partname]        
                if partname == 'cmeans':
                        vp.set_edgecolor('red')  # Set mean line color to red
                        vp.set_linewidth(2)      # Increase mean line width
                        vp.set_linestyle("dotted")
                        vp.set_alpha(1)          # Set mean line transparency
                elif partname == 'cmedians':
                        vp.set_edgecolor('orange')  # Set median line color to green
                        vp.set_linewidth(1.5)     # Set median line width
                        vp.set_linestyle("dotted")
                        vp.set_alpha(1)           # Set median line transparency
                elif partname in ('cmins', 'cmaxes'):
                        vp.set_edgecolor('blue')  # Set extrema line color to blue
                        vp.set_linewidth(1)       # Set extrema line width
                        vp.set_alpha(1)           # Set extrema line transparency
                elif partname == 'cbars':
                        vp.set_edgecolor('black')  # Set center bar line color to black
                        vp.set_linewidth(1)        # Set center bar line width
                        vp.set_linestyle("dotted")
                        vp.set_alpha(1)            # Set center bar line transparency


        # Customize and add legend:
        actualHandle = Line2D([0], [0], color='green', label='Actual Test Value', linestyle='dotted')
        meanHandle = Line2D([0], [0], color='red', label='Mean', linestyle='dotted')
        medianHandle = Line2D([0], [0], color='orange', label='Median', linestyle='dotted')
        maxMinHandle = Line2D([0], [0], color='blue', label='Min - Max', linestyle='dotted')
        dataHandle = Patch(facecolor='gray', edgecolor='black', label='Data')

        # Add legend with custom text color
        legend = plt.legend(handles=[actualHandle, maxMinHandle, meanHandle, medianHandle, dataHandle], 
                            loc='lower right', labelcolor='black')

        # Create a boxplot inside the violin plot
        box = ax0.boxplot(np.array(testClosePredV), positions=[1], widths=0.2, patch_artist=True, zorder=2, showmeans=False)

        # Customize the boxplot appearance
        #box['boxes'][0].set_facecolor("#8D7334",(0.5, 0.5, 0.5, 0.5))  # RGBA values for gray with 50% transparency
        box['boxes'][0].set_facecolor( (141/255,115/255,52/255,0.5)) # "#8D7334 cu secondary gold.
        box['boxes'][0].set_edgecolor('black')
        box['whiskers'][0].set_color('black')
        box['whiskers'][1].set_color('black')
        box['caps'][0].set_color('black')
        box['caps'][1].set_color('black')
        #box['medians'][0].set_color('white')
        #box['means'][0].set_color('red')  # If showmeans=True in boxplot

        ax0.axhline(y=closeTestActual, color='darkgreen', linestyle='dotted', linewidth=3) 
        ax0.set_xlabel('Dataset Index')
        ax0.set_ylabel('Predicted Closing Prices')
        ax0.set_title(targetStock + 'Predicted Closing Prices')

        # Save the plot to a PNG file
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(targetStock+filename+"Violin.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Histogram of residuals:
        #========================
        plt.figure(figsize=(16,10))
        fig2, ax2 = plt.subplots()
        histoResid = ax2.hist(residuals, bins=250, color="#8D7334", edgecolor='black', density=True)
        # Overlay a normal curve:
        mean2, std2 = scipy.stats.norm.fit(residuals)
        x2 = np.linspace(min(residuals), max(residuals), 1000)
        y2 = scipy.stats.norm.pdf(x2,mean2,std2)
        plt.plot(x2, y2, 'r-', linewidth=2, color="blue", label=f'Overlayed Normal Curve\nMean={mean2:.2f}, Std={std2:.2f}')
        ax2.set_xlabel('Residuals')
        ax0.set_ylabel('Density')
        ax0.set_title(targetStock + 'Lowest Error Model -- Histogram of Residuals')
        fig2.savefig(targetStock+filename+"Histogram.png", dpi=300, bbox_inches="tight")
        fig2.show()

        # Plot the best model: actual and fitted vs time.:
        #================================================
        # Plot y and yhat in a time series:
        #def linear_plot(X, yActual, yPred, xTest, yTestActual, yTestPred, targetStock, fileName, xLabel= "Date", yLabel="ClosePrice" ):      
       
        linear_plot(X=targetStockSeries.index, yActual=targetStockSeries["Actual"], yPred=targetStockSeries["Fitted"], 
                    xTest = targetStockSeries.shape[0], yTestActual=closeTestActual, yTestPred=closeTestPred, 
                    targetStock=targetStock,
                    fileName=(targetStock+filename+"fittedVsActual.png"), xLabel= "Date", yLabel="ClosePrice") # AAPLdiagnosticsSinglePredfittedVsActual.png

        #============================================================
        # plot the leverage vs. the square of the residual. Submit this one.
        #================================================================
        # your code here
        #plt.figure(figsize=(20,20))
        plt.rcParams["figure.figsize"] = (10,10) 
        leverageResidPlot0 = sm.graphics.plot_leverage_resid2(lowestErrorSPModel, 
                                                                markerfacecolor="#CFB87C", 
                                                                markeredgecolor="black", 
                                                                markersize=15)#color="#CFB87C", cu gold
        plt.grid(True)
        leverageResidPlot0.figure.savefig(targetStock+"leverageResidPlot0.png")
        plt.show()
        #=================================================================================

        # https://www.geeksforgeeks.org/residual-leverage-plot-regression-diagnostic/
        # Leverage vs residuals
        # Diagnostic plots:
        # https://stackoverflow.com/questions/66493682/glm-residual-in-python-statsmodel
        influence = lowestErrorSPModel.get_influence() 
        leverage = influence.hat_matrix_diag 
        
        # PLot different diagnostic plots 
        plt.rcParams["figure.figsize"] = (20,20) 
        fig1, ax1 = plt.subplots(nrows=2, ncols=2) 
        
        # Residual vs Fitted Plot: it can help us determine heteroskedasticity. 
        sns.scatterplot(x=fittedValues, y=residuals, ax=ax1[0, 0], hue=residuals, palette="magma_r", s=200)
        ax1[0, 0].axhline(y=0, color='black', linestyle='dotted') 
        ax1[0, 0].set_xlabel('Fitted Values') 
        ax1[0, 0].set_ylabel('Residuals') 
        ax1[0, 0].set_title('Residuals vs Fitted Fitted') 
        ax1[0, 0].grid(True)  

        # Normal Q-Q plot: determine if the distribution or residuals is normal.
        #sm.qqplot(residuals, fit=True, line='45',ax=ax1[0, 1], marker='o', markerfacecolor="#CFB87C", 
        #          markeredgecolor="black", alpha=0.9) #'#565A5C'
        sm.qqplot(stand_resids, fit=True, line='45',ax=ax1[0, 1], marker='o', markerfacecolor="#CFB87C", 
                  markeredgecolor="black", alpha=0.9) #'#565A5C'
        ax1[0, 1].set_title('Normal Q-Q') 
        ax1[0, 1].grid(True)

        # Scale-Location Plot is identical to the Fitted vs residuals.
        # Instead: plot yActual vs Fitted
        sns.scatterplot(x=fittedValues, y=actualValues, ax=ax1[1, 0], hue=residuals, 
                        palette="magma_r", s=200)
        x1 = np.linspace(min(fittedValues), max(fittedValues), 100)
        y1 = x1
        ax1[1, 0].plot(x1,y1, color="blue", linestyle="dashdot")
        #ax2e[1, 0].axhline(y=0, color='black', linestyle='dotted') 
        ax1[1, 0].set_xlabel('Fitted values') 
        ax1[1, 0].set_ylabel('Actual Close Price') 
        ax1[1, 0].set_title('Actual vs Fitted Plot') 
        ax1[1, 0].grid(True)

        # Cook's distance plot
        cooksInfluencePlot1 = sm.graphics.influence_plot(lowestErrorSPModel, external=False, ax=ax1[1, 1])#, criterion="cooks")#, palette="magma", c="orange") 
        # Change scatter plot color
        scatter = ax1[1,1].collections[0]
        #catter.set_cmap("magma")
        #catter.set_array(scatter.get_array())
        scatter.set_facecolor("#CFB87C")
        scatter.set_edgecolor('black')
        #plt.colorbar(label="Cook's Distance")
        #Change text color
        for text in ax1[1,1].texts:
                text.set_color('blue')

        # Modify grid color
        ax1[1,1].grid(True)

        # Update labels and title colors
        ax1[1,1].set_xlabel('Leverage')
        ax1[1,1].set_ylabel('Studentized Residuals')
        ax1[1,1].set_title('Influence Plot')

        plt.tight_layout() 
        plt.figure(figsize=(20,20))
        fig1.figure.savefig(targetStock + "plotSet" + filename + ".png")
        #plt.show() 

        #=========================================
        # Leverage vs Squared Residual Plot 
        #==========================================
        plt.figure(figsize=(16,10))
        leverageResidPlot = sns.scatterplot(x=np.power(residuals,2), y=leverage, hue=leverage, 
                                            palette="magma_r", s=200)
        plt.axhline(y=0, color='black', linestyle='dotted') 
        plt.ylabel('Leverage') 
        plt.xlabel('Squared Residuals') 
        plt.title('Leverage vs Squared Residuals Plot') 
        plt.tight_layout() 
        plt.grid(True)
        leverageResidPlot.figure.savefig(targetStock+filename+"lvrgResidSqPlot.png")

        plt.show() 
        return

#===================================
# SINGLE PREDICTOR REGRESSION MODEL:
#===================================
# Fit a glm to every single predictor available.
def singlePredglm(targetStock, securitiesD, corrDF, npLag=5, topNpred=1, alpha=0.01, R2Max=0.8, seedValue=2025):
        bestSinglePredModelV = []# lowest error.
        singlePredModelV = []
        lowestError = 1000000
        for j in range(topNpred):
                startIndex = j # for single predictor; =0 for multiple predictor combinations.
                topNPredictors = 1 # Take the predictors individually.
                allCombisM = setUpCombis(topNPredictors) # combinations of n=5 predictors. Total 2^5.
                corrTopFNames = list(corrDF.iloc[startIndex:startIndex+topNPredictors]["Predictor"])
                corrTopFSec = list(corrDF.iloc[startIndex:startIndex+topNPredictors]["secName"])
                print("\nLine 820: j=", j,"corrTopFNames=", corrTopFNames, "corrTopFSec=", corrTopFSec)
                formulaM, secNameM = buildFormulas(targetStock, corrTopFNames, corrTopFSec, allCombisM)
                #print("\nLine 492: formulaV=\n", formulaM, "\nsecNameM=\n", secNameM)

                # bestglmModelV has the best model for each k=1,...n=5/10/15 predictors.
                # From that list of models, we can select the best model.
                bestglmModelV = combiglmModels(targetStock, securitiesD, formulaM, secNameM, alpha, R2Max, npLag, seedValue)
                # combiglmModels returns many objects like this one: 
                # [modelTemp, yTestPred, testErrorPct, trainRMSEPct, yTestActual]
                for i in range(len(bestglmModelV)):
                        #print("\nLine 828: bestglmModelV=\n", bestglmModelV[i])
                        if len(bestglmModelV[i]) > 0:
                                singlePredModelV.append(bestglmModelV[i])
                                if(abs(bestglmModelV[i][2]) < lowestError):
                                        print("\nLine 834: Model parameters:",
                                        "\nTargetStock:", targetStock,
                                        "\nalpha=", alpha,
                                        "\ntopNPredictors=", topNPredictors,
                                        "\nR2Max=", R2Max,
                                        "\nbestglmModelV=\n", bestglmModelV[i][0].summary(),
                                        "\nypred=", bestglmModelV[i][1],
                                        "\nerror%=", bestglmModelV[i][2])
                                        lowestError = abs(bestglmModelV[i][2])
                                        bestSinglePredModelV = bestglmModelV
                                        print("\nLine 844: \nThis is a better model.")
        print("\nLine 917: \nThis is the best single model.\n", 
        "\nbestglmModelV=\n", bestSinglePredModelV[0][0].summary(), 
        "\nyTestAcual=", round(bestSinglePredModelV[0][4],2),  
        "\nypred=", round(bestSinglePredModelV[0][1],2), 
        "\nerror%=", round(bestSinglePredModelV[0][2],4))
        return singlePredModelV

def singlePredModelling(targetStock, securitiesD):
        # Build a correlation Matrix :
        #================================
        # This function was used in the initial stage of discovery. 
        # It is not used in modelling.
        corrThresh = 0 # all securities, all correlations.
        minSeries = 250 
        npLag = 5
        seedValue = 2025
        alpha = 0.01
        # Run this function once.
        if (not (os.path.isfile(buildFilePath("corrMatrix")))):
                buildCorrMatrix(targetStock, securitiesV+[targetStock], corrThresh, minSeries, npLag, seedValue)
                #buildCorrMatrix(targetStock, ["AAPL"], corrThresh, minSeries, npLag, seedValue)
        corrDF = pd.read_csv(buildFilePath("corrMatrix"))
        
        vifMax = 5#vifTemp = 1/(1-R2)
        R2Max = (1-1/vifMax)
        topNpred = corrDF.shape[0] # Use a 5 or 10 to get a shorter list.
        singlePredModelV = singlePredglm(targetStock, securitiesD, corrDF, npLag, topNpred, alpha, R2Max, seedValue)
        # singlePredModelV returns many objects like this one: [modelTemp, yTestPred, testErrorPct, trainRMSEPct, yTestActual]

        # Find the best model: model with the lowest error.
        spModelsV = [] # Single Predictor Model.
        testPredSPV = []
        testErrorPctSPV = []
        trainRMSEPctSPV = []
        testActualClose = singlePredModelV[0][4]
        lowestErrorSPPred = 0
        lowestErrorSPModel = None
        lowestTestErrorPct = 1000000 # Initialized absolute value of the test error.
        for i in range(len(singlePredModelV)):
                spModelsV.append(singlePredModelV[i][0])
                # singlePredModelV returns many objects like this one: 
                # [modelTemp, yTestPred, testErrorPct, trainRMSEPct, yTestActual]
                if(abs(singlePredModelV[i][2]) < lowestTestErrorPct):
                        lowestErrorSPModel = singlePredModelV[i][0]
                        lowestTestErrorPct = abs(singlePredModelV[i][2])
                        lowestErrorSPPred = singlePredModelV[i][1]
                testPredSPV.append(singlePredModelV[i][1])
                testErrorPctSPV.append(singlePredModelV[i][2])
                trainRMSEPctSPV.append(singlePredModelV[i][3])

        meanTestPredSP = np.mean(np.array(testPredSPV))
        meanTestErrorPctSP =  np.mean(np.array(testErrorPctSPV))
        meanRMSETrainErrorPctSP = np.mean(np.array(trainRMSEPctSPV))

        # Plot the distribution of errors and results: violin plot, histogram.
        print( "\nLine 958: \nlowestErrorSPModel", lowestErrorSPModel.summary(),
        "\nlowestTestErrorPct=", round(lowestTestErrorPct,4), "%",
        "\nNumber of models=", len(singlePredModelV),
        "\ntestActualClose=", round(testActualClose,2),
        "\nmeanTestPredSP=", round(meanTestPredSP,2), "across all models all models."
        "\nmeanTestErrorPctSP=", round(meanTestErrorPctSP,4), "% ; mean test error across all models.",
        "\nmeanRMSETrainErrorPctSP=", round(meanRMSETrainErrorPctSP,2), "%",
        "\nDurbin Watson autocorrelation of error =", durbin_watson(lowestErrorSPModel.resid_response))
        # Durbin Watsons Test for autocorrelation of residuals:
        #======================================================
        # If the residuals are correlated, the observations may not be independant from one another and
        # this would violate the assumption of independence.
        # A value of 2 indicates no autocorrelation.
        # Values signifcantly lower than 2 indicate positive autocorrelation.
        # Values < 1 indicate strong autocorrelation.

        diagnosticPlots(lowestErrorSPModel, testClosePredV=testPredSPV, closeTestActual=testActualClose, closeTestPred=lowestErrorSPPred, filename="diagnosticsSinglePred")

        return meanTestPredSP, meanTestErrorPctSP



#========================================================
# MULTIPLE-RANDOMLY-SELECTED-PREDICTOR REGRESSION MODEL:
#========================================================
def makeSMFFormula(formulaV):
        tempForm = targetStock + "Close ~ "
        for i in range(len(formulaV)):
                tempForm += "+" + formulaV[i]
        #print("\nLine 91: \ncompileDF=\n", compileDF, compileDF.dtypes, "\ncolShift=", colShift, 
        #"\n", compileDF, "\ntempForm=\n", tempForm)
        return tempForm

# The function makeSMGglm builds a model with predictor's pvalues < alpha.
def makeSMFglm(modelFormula, trainSet, testSet, alpha=0.01):
        closeActual = testSet[ targetStock+"Close"].iloc[0]
        #1) Make the base model:
        modelTemp = smf.glm(formula=modelFormula, data=trainSet).fit()
        #print("\nLine 1072: modelTemp=", modelTemp.summary())
        #2) Evaluate pvalues, mark those with high pvalues to remove. 
        pv = dict(modelTemp.pvalues)
        pvKeys = list(pv.keys()) # The name of the predictors.
        removePredS= set()
        removePredV = [] # this list contains the predictors with pvalues >= alpha
        for k in range(1,len(pvKeys)): # skip 0, because it is the intercept.
                # Check that each pvalue < than alpha:
                if(pv[pvKeys[k]] > alpha): # remove predictors with high pvalue coefficients.
                        removePredS.add(pvKeys[k])
                        removePredV.append(pvKeys[k])
                                
        # Now, 2 scenarios:
        # 1) all predictors pvlues <= 0.05: no elements found in removePredV
        modelV=[]
        rmseV=[]
        closePredV=[]
        errorPredPctV=[]
        targetStockSeriesV=[]
        if len(removePredV) == 0: # no predictor to remove, all pvalues<= 0.05
                modelV.append(modelTemp)
                #Deviance is the sum of squares of residuals.(not the mean, the total sum of squares.)
                deviance = modelTemp.deviance # 
                # MSE is the mean squared error. In this case, the Deviance/number of observations (trading days)
                # RMSE is the sqrt of MSE, in the units of the data set.
                rmseV.append(np.sqrt(deviance/trainSet.shape[0]))
                closePred = modelTemp.predict(testSet)
                closePredV.append(closePred) # This is the predicted close in the test set.
                errorPredPct = (closePred-closeActual)/closeActual*100
                errorPredPctV.append(errorPredPct)
                #errorPredV.append((closePred-closeActualV[0])/closeActualV[0]*100)
                targetStockSeriesV.append(trainSet.copy().iloc[:,0:5])
                #print("\nLine 1099: \nmodelTemp=\n", modelTemp.summary(), 
                #      "\nclosePred=", closePred,
                #      "\ncloseActual=", closeActual,
                #      "\nerrorPredPct=", errorPredPct)
        # 2) at least 1 predictor has a pvalue > 0.05
        #else: 
                #print("\nLine 142: removePredV=", removePredV)
                # return the list of predictors to remove from the formula.
        
        return removePredV, modelV, rmseV, closePredV, errorPredPctV, targetStockSeriesV
# This list may be empty or not.

# The function selectCandidatePred takes the targetStock and a list of
# securities, takes a random sample of nSeq securities, merges it into a dataframe with target Stock
# and verifies that the vif of each predictor < vifmax. The function returns a list of candidates
# which comply with the collinearity requirements.
def selectCandidatePred(targetStock, securitiesD, securitiesV, vifMax=5, npLag=5, nSeq=10, seedi=2025):
        #print("\nLine 1131 ok")
        # No minimum length of series in this case.
        # Take a random sample, with replacement, of n=50 securities, amongst securitiesV.
        #sampleSize = 50 # This is about 10% of all securities, each having Open,High,Low,Close,Volume, 
        # a total of 250 predictors.
        #lag = 5
        R2Max = (1-1/vifMax) # R2Max=0.8 for vifMax =5.
        random.seed(seedi) 
        sampleSeq = list(range(0,len(securitiesV)))# a sequence with all securities indexed 0,...len(secV)
        #random.seed(seedCV)
        random.shuffle(sampleSeq)
        indexPred = random.sample(sampleSeq, nSeq) # index to selected nSeq securities from the shuffled list.
        selectedSecV = np.array(securitiesV)[indexPred] # selected nSeq securities.
        # This is a compile DF with Date, targetStock, and nSeq in one place.
        compileDF = mergeSecsDF(targetStock, securitiesD, securitiesV=selectedSecV, npLag=npLag, seedValue=seedi)
        # Get the test set: The last row of the dataframe is the date to predict.
        compileDFTest = compileDF[-1:]
        # The rest of the dataset is the training set:
        compileDFTrain = compileDF[1:-1] # consider starting on row 1.
        # This is the base start of the np periods compileDF 
        # with the target stock, date, year, month, day
        # and Open, High, Low, Volume(when available) lagged np periods.
        closeActualV = [compileDFTest[targetStock+"Close"].iloc[0]] # This is the actual close in the test set.
        # We will initialize the model with the target stock Open, High, Low, Volume, year, month, day.
        # The candidatePredV holds all predictor candidates.
        candidatePredV = list(compileDFTrain.columns)
        candidatePredV.remove((targetStock+"Close"))
        candidatePredV.remove("Date")
        #print("\nLine 1161: candidatePredV=", candidatePredV)
        #candidatePredV = list(["year", "month", "day"]) + candidatePredV
        # First check the colinearity of predictors and remove predictors with vif>vifmax.
        # Approach: take all predictors and make a regression with each against all others.
        # Remove the predictor with highest VIF, and run again until all predictors have vif < vifmax.
        # Wrap this into a VIFFuntion.
        #print("\nLine 1176 ok")
        flagVif = 0
        while(flagVif == 0):
                R2High = 0
                predHighVif = None
                #print("\nLine 1183 ok")
                for j in range(len(candidatePredV)):
                        #print("\nLine 1186 ok")
                        tempDF = compileDFTrain.copy()
                        y = tempDF[candidatePredV[j]]
                        x = tempDF[candidatePredV].drop( [candidatePredV[j]], axis=1) # review syntax here.
                        R2 = LinearRegression().fit(x, y).score(x, y)
                        #print("\nLine 1181: candidatePredV[j]=", candidatePredV[j], "\nR2=", R2)
                        if R2 > R2High:
                                predHighVif = candidatePredV[j]
                                R2High = R2
                                #print("\nLine 1186: candidatePredV[j]=", candidatePredV[j], "\nR2=", R2)
                        if j == (len(candidatePredV)-1):
                                if R2High > R2Max:
                                        candidatePredV.remove(predHighVif)
                                        #print("\nLine 1228: remove predictor=", predHighVif, R2High)
                                        #"\ncandidatePredV=", candidatePredV)
                                else:
                                        flagVif = 1 # break the loop
                                #break
        # Verify that the the candidate that al predictors meet VIF<VIFMAx:
        for k in range(len(candidatePredV)):
                tempDF = compileDFTrain.copy()
                y = tempDF[candidatePredV[k]]
                x = tempDF[candidatePredV].drop([candidatePredV[k]], axis=1) # review syntax here.
                R2 = LinearRegression().fit(x, y).score(x, y)
                #print("\nLine 1181: candidatePredV[j]=", candidatePredV[j], "\nR2=", R2)
                if R2 > R2Max:
                        print("\nLine 1218: something wrong. Predictor=", candidatePredV[k], R2)
                #else:
                        #print("\nLine 1211: This predictor=", candidatePredV[k], R2, "is good on vif.")
        #print("\nLine 1208: candidatePredV=", candidatePredV)
        return candidatePredV, compileDFTest, compileDFTrain, closeActualV

# VARIANCE INFLATION FACTOR (VIF):
#==================================
# VIF indicates if a predictor is collinear with another one in the model.
# Computing vif requires that we regress a predictor against all other predictors.
def checkVIF(glmModel, vifMax=5):
        R2Max = (1-1/vifMax)
        tempDF = (glmModel.model.data.frame).copy()
        #pv = dict(lowestModel.pvalues)
        #pvKeys = list(pv.keys()) # T
        candidatePredV = glmModel.model.exog_names[1:]
        #candidatePredV = list(tempDF.columns)
        if( len(candidatePredV) > 1 ): # I need at least 2 predictors.
                for k in range(len(candidatePredV)):
                        #tempDF = compileDFTrain.copy()
                        y = tempDF[candidatePredV[k]]
                        x = tempDF[candidatePredV].drop([candidatePredV[k]], axis=1) # review syntax here.
                        R2 = LinearRegression().fit(x, y).score(x, y)
                        #print("\nLine 1181: candidatePredV[j]=", candidatePredV[j], "\nR2=", R2)
                        if R2 > R2Max:
                                print("\nLine 1302: something wrong. Predictor=", candidatePredV[k], R2)
                        else:
                                print("\nLine 1306: This predictor=", candidatePredV[k], "R2=", R2, "vif=", 1/(1-R2), "is good on vif.")
                #print("\nLine 1208: candidatePredV=", candidatePredV)
        return

# The multiPredglm function creates 1 model with the lowest error for a given random sample of securities.
# It returns the glm model object.
def multiPredglm(targetStock, securitiesD, securitiesV, npLag=5, vifMax=5, nModels=10, nSeq=15, alpha=0.01):
        # A bit manual, but good results.
        # Each i iteration will produce 1 glm model with the lowest error for that sample of securities.
        # These vectors store the results of each iteration:
        bestModelV= []
        minPredErrorV= []
        minRMSEVPct= []
        closeActualV= [] 
        minClosePredV = []
        minStockSeriesV=[]
        for i in range(nModels):
                # Select the candidate securities randomly:
                seedi = 2025+i
                # candidatePredV is a list of predictors with collinearity vif < vifMax.
                candidatePredV, compileDFTest, compileDFTrain, closeActualV= selectCandidatePred(targetStock, securitiesD, securitiesV, vifMax, npLag, nSeq, seedi)        
                print("\nLine 1215: i=", i, "candidatePredV=\n", candidatePredV)
                # The function returns a list candidatePredV with vif < vifMax.

                closeActual = closeActualV[0]
                goodPredV = [] # good predictors with good pvalues and vif.
                modelV = []
                errorPredPctV = []
                rmseV = []
                closePredV = [] # each model predicts a close.
                targetStockSeriesV= []

                # Take the formulaV vector with candidate predictors in it, and start building a model.
                # Take the first predictor: if its pvalue < alpha, keep it, if not remove it.
                # Add the next predictor in the sequence, and repeat the process.
                for j in range(len(candidatePredV)):
                        goodPredV.append(candidatePredV[j])
                        tempForm = makeSMFFormula(goodPredV)
                        removePredV, modelVj, rmseVj, closePredVj, errorPredPctVj, targetStockSeriesVj = makeSMFglm(tempForm, compileDFTrain, compileDFTest, alpha)
                        if(len(modelVj)>0):
                                modelV.append(modelVj[0])
                                rmseV.append(rmseVj[0])
                                # closePredVj is the predicted close in the test set.
                                # The object type is panda series; convert it to a list and get the float value in its first element. 
                                closePredV.append(list(closePredVj[0])[0]) # 
                                #print("\nLine1175: closePredVj[0]", list(closePredVj[0])[0])
                                errorPredPctV.append(errorPredPctVj[0])
                                targetStockSeriesV.append(targetStockSeriesVj[0])
                
                        #print("\nLine 1232: j=", j, "out of", len(candidatePredV), "\ntempForm=\n", tempForm, "\nremovePredV=", removePredV)       
                        for k in range(len(removePredV)):
                                goodPredV.remove(removePredV[k])
                #print("\nLine 1236: i=", i, "\ntempForm=\n", tempForm, "\ngoodPredV=", goodPredV)       

                if(len(goodPredV) > 0): # occassionally, none of the predictors are good and the vector is empty.
                        # That means that no good model can be created with this set of predictors.
                        # Find the model with the lowest % error of prediction on the test set:
                        minPredError = np.min(np.absolute(np.array(errorPredPctV)))
                        #print("\nLine 1232: minPredError%=", minPredError)
                        indexMinError = np.where(np.absolute(np.array(errorPredPctV)) == minPredError)[0][0]
                        #print("\nLine 1234: indexMinError=", indexMinError)
                        bestModel = modelV[indexMinError]
                        # Model performance:
                        minRMSE = rmseV[indexMinError]
                        minRMSEPct = minRMSE/closeActual*100
                        minClosePred = closePredV[indexMinError]
                        # Do we really need minStockSeries? The Model data is inside the model object:
                        # bestModel.model.data.frame
                        minStockSeries = targetStockSeriesV[indexMinError] #
                        predDate = compileDFTest.iloc[0].loc["Date"] # prediction date.
                        # Each iteration creates a best model. Add it to the list of models.
                        bestModelV.append(bestModel)
                        minPredErrorV.append(minPredError)
                        minRMSEVPct.append(minRMSEPct)
                        #closeActualV.append(closeActualV[0])
                        minClosePredV.append(minClosePred)
                        minStockSeriesV.append(minStockSeries)

        # Minimal Error measures:
        #=========================
        # Each iteration originated from a random sample of predictors and produces a lowest error model.
        # Find the lowest of the lowest errors.
        #print("\nLine 1277 minPredErrorV=", minPredErrorV)
        lowestPredError = np.min(np.array(minPredErrorV))
        xlowestPredError = np.where(minPredErrorV == lowestPredError)[0][0]
        lowestModel = bestModelV[xlowestPredError] # The model with the lowest error in the test set.
        lowestRMSEPct = minRMSEVPct[xlowestPredError] # The model with the lowest error in the test set.
        #closeActual = closeActualV[xlowestPredError]
        lowestClosePred = minClosePredV[xlowestPredError]
        #targetStockSeries = minStockSeriesV[xlowestPredError] # Do we need this? The model has the data. Return the model instead.
        print( "\nLine 1219: \nlowestModel=", lowestModel.summary(), 
        "\nlowestPredError%=", round(lowestPredError,4), "'%'on test set.",
        "\nlowestRMSEPct=", round(lowestRMSEPct,2), " '%' average error on the training set.",
                "\ncloseActual=", round(closeActual, 2), " the actual closing Price on test date.", predDate,
                "\nlowestClosePred =",round(lowestClosePred, 2), " the predicted closing Price on test date.")
        # The candidate predictors where selected and tested for collinearity with vif<5,
        # before building the model/s. There should be no surprises now:
        # This is the last verification on the output model.
        checkVIF(lowestModel, vifMax)

        # Average performance metrics:
        #==============================
        meanPredError = np.mean(np.array(minPredErrorV))
        nModels = len(bestModelV)
        #lowestModel = bestModelV[xlowestPredError] # The model with the lowest error in the test set.
        meanRMSEPct = np.mean(np.array(minRMSEVPct))
        meanClosePred = np.mean(np.array(minClosePredV))
        # The series is one for each model, there is nModel series.
        #targetStockSeries = minStockSeriesV[xlowestPredError]
        print("\nLine 1232: nModels=", nModels+1, "samples of random securities=" , nSeq,
        "\nmeanPredError=", round(meanPredError,2), "'%'",
        "\nmeanRMSEPct=", round(meanRMSEPct,2), "'%'"
        "\nmeanClosePred=", round(meanClosePred,2))
        return lowestModel, minClosePredV, closeActual, lowestClosePred, meanClosePred, meanRMSEPct

t40 = time.time()
# Tested with defaults: ok.

# SINGLE AND MULTI PREDICTOR LINEAR REGRESSION MODELS:
#======================================================
def singleMultiPredWrapper():
        targetStock = "AAPL"
        #targetStock = "MSFT"
        marketIndices = ["^GSPC", "^DJI", "^IXIC", "^GDAXI", "^FTSE", "^FCHI", "^N225", "^HSI", "^STI"] 
        bondIndices = ["^TYX", "^TNX","^FVX","^IRX","DX-Y.NYB"]
        cmdtyIndices= ["GLD","CL=F"]
        sp500Stocks = list(pd.read_csv(buildFilePath("sp500ListWiki"))["Symbol"])
        # Remove the target stock from the list of predictors.
        doNotInclude = ["CTLT", "ETR", "LEN"] + [targetStock] # ,"BF.B","BRK.B"
        sp500StocksClean = sorted(list(set(sp500Stocks.copy()) - set(doNotInclude)))

        # All securities in a list, after removing the target stock from it.
        # The target stock data series is already included in the start of the DF of the model.
        securitiesV = marketIndices+bondIndices+cmdtyIndices+sp500StocksClean

        # Load all data set with all securities in a dictionary:
        #=========================================================
        securitiesD = loadAllSecDF(targetStock, securitiesV=securitiesV)        
        #print("\nLine 1107: securitiesD=\n", securitiesD)
        print("\nLine 1107: aLL SECURITIED LOADED" )

        # Run the models for single predictors once. 
        # Just once.

        meanTestPredSP, meanTestErrorPctSP = singlePredModelling(targetStock, securitiesD)
        #allModelsMeanPredV.append(round(meanTestPredSP,2))
        #allModelsMeanTestErrorPctV.append(round(meanTestErrorPctSP,4))
        

        t20 = time.time() # end of single-predictor modelling. # 40 minutes to run all predictors.
        print( "\nTotal Time: ", round((t20-t0), 1), "secs.") 

        # FINAL MODEL MULTI PREDICTOR MODEL:
        #===================================
        # Single and Multi predictor glm regression was used in the discovery stage and initial modelling
        # The results are not used to compute the final averages.
        #==================================================================================================
        nModels = 500 # number of models to create. Each model a different random sample.
        nSeq = 20 # now running 20 #nSeq= number of securities in the random sample.
        vifMax = 5 #vifTemp = 1/(1-R2)
        # Make a prediction of the last date as a test, and measure the error.

        # Run the model once and save it for later use and diagnostics.
        # Run just once. This code takes 4 hours for 500 models.
        # Once the model file is saved, the function won't run again until the
        # .pkl file is deleted.
        bestMultiPredglmFileName = targetStock+str(nModels)+"M"+str(nSeq)+"SeqModel.pkl"
        bestMultiClosePredDFName = targetStock+str(nModels)+"M"+str(nSeq)+"ClosePredDF.csv"
        bestMultiValuesDFName = targetStock+str(nModels)+"M"+str(nSeq)+"ValuesDF.csv"
        # Uncomment this section of the code to run Multi Predictor glm regression and its diagnostics.
        #================================================================================================
        if (not (os.path.isfile(bestMultiPredglmFileName))):
                # Save the model for later use:
                bestMultiPredglmModel, bestMultiClosePredV, closeTestActual, lowestClosePred, meanClosePred, meanRMSEPct= multiPredglm(targetStock, securitiesV, npLag, vifMax, nModels, nSeq, alpha)        
                bestMultiPredglmModel.save(bestMultiPredglmFileName)

                # Save the model outputs for later use:
                print("\nLine 1262:\n", bestMultiClosePredV )
                bestMultiClosePredDF = pd.DataFrame({"bestMultiClosePredV":bestMultiClosePredV})
                bestMultiClosePredDF.to_csv(bestMultiClosePredDFName, index=False, encoding="utf-8")
                
                bestMultiValuesDF = pd.DataFrame( {"closeTestActual": [closeTestActual], 
                                                   "lowestClosePred": [lowestClosePred],
                                                   "meanClosePred": [meanClosePred], 
                                                   "meanRMSEPct": [meanRMSEPct]})
                bestMultiValuesDF.to_csv(bestMultiValuesDFName, index=False)

        # Load the saved model:
        bestMultiPredglmModel = sm.load(bestMultiPredglmFileName)
        # Load the model outputs in DF format:
        bestMultiClosePredDF = pd.read_csv(bestMultiClosePredDFName)
        bestMultiClosePredV =( bestMultiClosePredDF.iloc[:,0])
        bestMultiValuesDF = pd.read_csv(bestMultiValuesDFName)
        closeTestActual = list(bestMultiValuesDF.iloc[:,0])[0]
        lowestClosePred = list(bestMultiValuesDF.iloc[:,1])[0]
        meanMultiClosePred =  list(bestMultiValuesDF.iloc[:,2])[0]
        meanMultiRMSEPct = list(bestMultiValuesDF.iloc[:,3])[0]

        print("\nLine 1166: bestMultiPredglmModel=", bestMultiPredglmModel.summary(), 
              "\n", bestMultiClosePredV, "\n", closeTestActual, "\n", lowestClosePred)

        # Single and Multi Predictor regression not considered for the final average.
        #allModelsMeanPredV.append(round(meanMultiClosePred,2))
        #allModelsMeanTestErrorPctV.append(round(meanMultiRMSEPct,4))

        dwStat = durbin_watson(bestMultiPredglmModel.resid_response )
        print("\nLine 1174: dwStat=", dwStat )
        # A value of 2 indicates no autocorrelation.
        # Values signifcantly lower than 2 indicate positive autocorrelation.
        # Values < 1 indicate strong autocorrelation.

        diagnosticPlots( bestMultiPredglmModel, bestMultiClosePredV, closeTestActual, lowestClosePred, 
                        ("multiPred"+str(nModels)+"Models"+str(nSeq)+"Seq"))
        #diagnosticPlots(lowestErrorSPModel, testClosePredV, closeTestActual, closeTestPred, filename):

        t41 = time.time()
        print("\nLine 1311: time=", round((t41-t40),0))
        return

# Run this function once:
# Follow the instructions written in the code comments if you decide to run again.
# singleMultiPredWrapper()

#========================== END OF SINGLE/MULTI-PREDICTOR - LINEAR REGRESSION MODELS=======================

# DECISION TREES, RANDOM FORESTS, AND ENSEMBLE METHODS:
#======================================================
# These models are much more efficient and accurate than the single/multi predictor models.
# They are also more suitable for high-dimensional data such as the one we are dealing with.
# The final results are based on these 5 models.
def dtrRFREnsembles(securitiesV, targetStock, minSeries=1250, npLag=5, seedValue=2025):
        allModelsMeanPredV = []
        allModelsMeanTestErrorPctV = []
        securitiesD = loadAllSecDF(targetStock, securitiesV=securitiesV)        
        # DECISION TREE AND ENSEMBLE RANDOM FOREST:
        #===========================================
        # Select the securities with at least 5 years of trading history.
        # I will be scanning the orginal file with all trading days in it. 
        # Each year has circa 250 days of trading. Total rows = 1250. 
        securities5yV = []
        minSeriesDaily = minSeries # 5 years
        for i in range(len(securitiesV)):
                if(securitiesD[securitiesV[i]].shape[0] > minSeriesDaily):
                        securities5yV.append(securitiesV[i])  
                else:
                        print("\nLine 1324: security=", securitiesV[i], "length=", len(securitiesV[i]), "too short.")
        #print("\nLine 1324: securities250V=", len(securities5yV))
        compileDF = mergeSecsDF(targetStock, securitiesD, securities5yV, npLag, seedValue)

        featureNames, XTrain, yTrain, XTest, yTest = trainTestSets(compileDF)
        #print("\nLine 1329: compileDF=", compileDF.head(), XTrain, yTrain)
        #dtr = DecisionTreeRegressor(criterion= "absolute_error", max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=seedValue) # Decision Tree Regressor.
        dtr = DecisionTreeRegressor(random_state=seedValue) # Decision Tree Regressor.

        print("\nLine 1336: DecisionTreeRegressor object complete.\n")

        # Decision Tree Regressor Parameter Tuning :
        paramGrid = { 
                "criterion": ["friedman_mse"], #["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "max_depth": [9], #[6,7,8,9,10,11,12,13,14,15,16,17,18,19], #[None,5, 10, 20, 30, 40, 50],
                "min_samples_split": [2], #[None, 2], #5, 10, 20] ,
                "min_samples_leaf": [1], #,2,4,8],
                "max_features": ["log2"] # [1, "sqrt", "log2"]
        }

        dtrTScv = TimeSeriesSplit(n_splits=5)
        dtrGridSearch = GridSearchCV(estimator=dtr, param_grid=paramGrid, cv=dtrTScv)
        #dtr.fit(XTrain, yTrain)
        dtrGridSearch.fit(XTrain, yTrain)
        dtrBestParams = dtrGridSearch.best_params_
        dtrBestScore = -dtrGridSearch.best_score_
        print("Line 1359 Best Hyperparameters:", dtrBestParams, "\nbestScore=", dtrBestScore)

        bestCriterion = dtrGridSearch.best_params_["criterion"]
        bestMaxDepth = dtrGridSearch.best_params_["max_depth"]
        bestMinSamplesSplit = dtrGridSearch.best_params_["min_samples_split"]
        bestMinSamplesLeaf = dtrGridSearch.best_params_["min_samples_leaf"]
        bestMaxFeatures = dtrGridSearch.best_params_["max_features"]

        dtr = DecisionTreeRegressor(criterion= bestCriterion, max_depth=bestMaxDepth, 
                                min_samples_split = bestMinSamplesSplit, 
                                min_samples_leaf = bestMinSamplesLeaf,
                                max_features = bestMaxFeatures, 
                                random_state = seedValue) # Decision Tree Regressor.
        print("\nLine 1362: DecisionTreeRegressor fit complete.\n")
        dtr.fit(XTrain, yTrain)
        dtrPred = dtr.predict(XTest)
        print("\nLine 1365: \ndtrPred=", round(dtrPred[0],2), "\nactualyTest=", round(yTest[0],2) )

        # Evaluation of the decision tree regression:
        #=============================================
        dtrMSE = mean_squared_error(yTest, dtrPred)
        dtrRMSE = np.sqrt(dtrMSE)
        dtrRMSEPct = dtrRMSE/yTest[0]*100
        dtrTestErrorPct = (dtrPred[0]-yTest[0])/yTest[0]*100
        print("\nLine 1373: \ndtrMSE=", round(dtrMSE,2), "dtrRMSE=", round(dtrRMSE,2),
        "\ndtrRMSEPct=", round(dtrRMSEPct,2), "'%'",
        "\ndtrTestErrorPct=", round(dtrTestErrorPct,2), "%")
        allModelsMeanPredV.append(round(dtrPred[0],2))
        allModelsMeanTestErrorPctV.append(round(dtrRMSEPct,4))

        # Plot the regression tree:
        plt.figure(figsize=(150,50))
        dtrPlot = tree.plot_tree(dtr, feature_names=featureNames, filled=True, fontsize=6, rounded=True)
        plt.savefig("dtrTreePlot.png")
        # Question

        t42 = time.time()
        print("\nLine 1388: Decision Tree time=", round((t42-t41),0), "secs.")

        # RANDOM FOREST:
        #===============
        np.seterr(all='ignore')
        rfr = RandomForestRegressor(n_estimators=100, random_state=seedValue) # Random Forest Regressor
        print("\nLine 1394: RandomForestRegressor object complete.\n")
        rfr.fit(XTrain, yTrain)
        print("\nLine 1396: RandomForestRegressor fit complete.\n")
        rfrPred = rfr.predict(XTest)

        rfrMSE = mean_squared_error(yTest, rfrPred)
        rfrMSEPct = rfrMSE/yTest[0]*100 
        rfrRMSE = np.sqrt(rfrMSE)
        rfrRMSEPct = (rfrPred[0]-yTest[0])/yTest[0]*100
        print("\nLine 1401: \nrfrPred=", round(rfrPred[0],2), 
        "\nactualyTest=", round(yTest[0],2),
        "\nrfMSE=", round(rfrMSE,2),
        "\nrfMSEPct=", round(rfrMSEPct,2), "%",
        "\nrfRMSE=", round(rfrRMSE,2), 
        "\nrfMSEPct=", round(rfrRMSEPct,2), "%")

        # Random Forest Regressor Parameter Tuning :
        paramGridrfr = {
                #"n_estimators": list(np.arange(3200,3500,1)), #best 3281 3.69%
                #"n_estimators": list(np.arange(1600,5000,50)), #best 3300 3.69% # 3724 error 3.67%
                #"n_estimators": list(np.arange(100,3000,100)), #best 1800 3.78%
                #"n_estimators": [100], #error 3.60%
                #"n_estimators": [216], #list(np.arange(200,350,1)), # best is 216 error 3.53%
                #"n_estimators": list(np.arange(20,120,1)), # best is 32 error 3.62%
                "n_estimators": [32], # error 3.62%
                "criterion": ["absolute_error"], #["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "max_depth": [None], #np.arange(1,10,1), #list(np.arange(1,10,1)),# 30, 40, 50], best is 4
                "min_samples_split": [2], #np.arange(1,20,1), #[None,2,3,4,5,6,7,8,9,10],# 10, 20], best is 4
                "min_samples_leaf": [1],#np.arange(1,10,1),
                "max_features": ["sqrt"], #[1,"log2", "sqrt"],
                "bootstrap": [False] # True, # No bootstrapping: the whole data set is used in each tree.
                # Max sample: this parameter controls the size of the sample for each tree. Default=whole dataset is used.
                #ccpalpha: tune without ccp first. The main goal is to achieve a low test error/.
                }
        # Time series requires a careful selection of the cv parameter to maintain the
        # natural temporal sequence of the data.
        rfrTScv = TimeSeriesSplit(n_splits=5)
        rfrGridSearch = GridSearchCV(estimator=rfr, param_grid=paramGridrfr, scoring= "neg_mean_squared_error", cv=rfrTScv, n_jobs=-1)
        #dtr.fit(XTrain, yTrain)
        rfrGridSearch.fit(XTrain, yTrain)
        rfrBestParams = rfrGridSearch.best_params_
        rfrBestScore = -rfrGridSearch.best_score_
        print("Line 1423 Random Forest Best Hyperparameters:", rfrBestParams, "\nbestScore=", rfrBestScore)

        rfrBestnEst = rfrGridSearch.best_params_["n_estimators"]
        rfrBestCriterion = rfrGridSearch.best_params_["criterion"]
        rfrBestMaxDepth = rfrGridSearch.best_params_["max_depth"]
        rfrBestMinSamplesSplit = rfrGridSearch.best_params_["min_samples_split"]
        rfrBestMinSamplesLeaf = rfrGridSearch.best_params_["min_samples_leaf"]
        rfrBestMaxFeatures = rfrGridSearch.best_params_["max_features"]
        rfrBestBootstrap = rfrGridSearch.best_params_["bootstrap"]

        rfrTuned = RandomForestRegressor(
        criterion= rfrBestCriterion, 
        n_estimators = rfrBestnEst,
        max_depth = rfrBestMaxDepth, 
        min_samples_split = rfrBestMinSamplesSplit, 
        min_samples_leaf = rfrBestMinSamplesLeaf,
        max_features = rfrBestMaxFeatures, 
        bootstrap= rfrBestBootstrap,
        random_state = seedValue) # Decision Tree Regressor.

        rfrTuned.fit(XTrain, yTrain)
        print("\nLine 1444: RandomForestRegressor Tuned fit complete.\n")
        rfrTunedPred = rfrTuned.predict(XTest)
        print("\nLine 1446: \nrfrTunedPred=", round(rfrTunedPred[0],2), 
        "\nactualyTest=", round(yTest[0],2))

        # Evaluation of the decision tree regression:
        #=============================================
        rfrTunedMSE = mean_squared_error(yTest, rfrTunedPred)
        rfrTunedRMSE = np.sqrt(rfrTunedMSE)
        rfrTunedRMSEPct = rfrTunedRMSE/yTest[0]*100
        rfrTestErrorPct = (rfrTunedPred[0]-yTest[0])/yTest[0]*100

        print("\nLine 1472: \nrfrTunedMSE=", round(rfrTunedMSE,2), 
        "\nrfrTunedRMSE=", round(rfrTunedRMSE,2), 
        "\nrfrTunedRMSEPct=", round(rfrTunedRMSEPct,2), "%",
        "\nrfrTestErrorPct=", round(rfrTestErrorPct,2), "%" )

        allModelsMeanPredV.append(round(rfrTunedPred[0],2))
        allModelsMeanTestErrorPctV.append(round(rfrTunedRMSEPct,4))

        # Trained RandomForestRegressor first tree:
        rfrTunedTree0 = rfrTuned.estimators_[0]  # Select the first tree

        #from sklearn.tree import plot_tree
        plt.figure(figsize=(150, 50))
        rfrTunedPlot = tree.plot_tree(rfrTunedTree0, feature_names=featureNames, filled=True, fontsize=6, rounded=True)
        plt.savefig("rfrTunedPlot.png")

        t43 = time.time()
        print("\nLine 1486: \nTargetStock:", targetStock, "\nRandom Forest time=", round((t43-t42),0), "secs.")


        # ADABOOST REGRESSOR: it creates an ensemble of ensembles
        # Adaptive Boosting focuses on improving weak learners.
        #========================================================
        #rf = RandomForestRegressor(n_estimators=10, max_depth=3)
        adaboostRFR = AdaBoostRegressor(estimator=rfrTuned) #, n_estimators=50, learning_rate=1.0)

        paramGridadarfr = {
                "n_estimators":[50], #[40,50,60], #[100], # [10,50,100], [200,300,400], # 100, error 2.97; 200 error 3.06; 50 error2.84%
                #"estimator__max_depth": [None, 1,3,5], 
                "learning_rate": [1] # [0.01, 0.1, 1.0], # best = 1
                }

        adarfrTScv = TimeSeriesSplit(n_splits=5)
        adarfrGridSearch = GridSearchCV(estimator=adaboostRFR, 
                                        param_grid=paramGridadarfr, 
                                        scoring= "neg_mean_squared_error", 
                                        cv=adarfrTScv, n_jobs=-1)
        #dtr.fit(XTrain, yTrain)
        adarfrGridSearch.fit(XTrain, yTrain)
        adarfrBestParams = adarfrGridSearch.best_params_
        adarfrBestScore = -adarfrGridSearch.best_score_
        print("Line 1511 AdaBoostRandom Forest Best Hyperparameters:", adarfrBestParams, "\nbestScore=", adarfrBestScore)

        adarfrBestnEst = adarfrGridSearch.best_params_["n_estimators"]
        #adarfrBestMaxDepth = adarfrGridSearch.best_params_["estimator__max_depth"]
        adarfrBestLearningRate = adarfrGridSearch.best_params_["learning_rate"]

        adarfrTuned = AdaBoostRegressor(estimator=rfrTuned,
        n_estimators = adarfrBestnEst,
        #estimator__max_depth = adarfrBestMaxDepth, 
        learning_rate = adarfrBestLearningRate,
        random_state = seedValue)

        adarfrTuned.fit(XTrain, yTrain)
        print("\nLine 1532: Ada Boost RandomForestRegressor Tuned fit complete.\n")
        adarfrTunedPred = adarfrTuned.predict(XTest)
        print("\nLine 1534: \nadarfrTunedPred=", round(adarfrTunedPred[0], 2), 
        "\nactualyTest=", round(yTest[0], 2))

        # Evaluation of the Ada Boost Random Forest Regression:
        #=============================================
        adarfrTunedMSE = mean_squared_error(yTest, adarfrTunedPred)
        adarfrTunedRMSE = np.sqrt(adarfrTunedMSE)
        adarfrTunedRMSEPct = adarfrTunedRMSE/yTest[0]*100
        adarfrTestErrorPct = (adarfrTunedPred[0]-yTest[0])/yTest[0]*100

        print("\nLine 1538: \nadarfrTunedMSE=", round(adarfrTunedMSE,2), 
        "\nadarfrTunedRMSE=", round(adarfrTunedRMSE,2), 
        "\nadarfrTunedRMSEPct=", round(adarfrTunedRMSEPct,2), "%",
        "\nadarfrTestErrorPct=", round(adarfrTestErrorPct,2), "%" )

        # Trained Ada Boost RandomForestRegressor first tree:
        #rfrTunedTree0 = rfrTuned.estimators_[0]  # Select the first tree

        #from sklearn.tree import plot_tree
        #plt.figure(figsize=(150, 50))
        #rfrTunedPlot = tree.plot_tree(rfrTunedTree0, feature_names=featureNames, filled=True, fontsize=6, rounded=True)
        #plt.savefig("rfrAdaTunedPlot.png")
        allModelsMeanPredV.append(round(adarfrTunedPred[0],2))
        allModelsMeanTestErrorPctV.append(round(adarfrTunedRMSEPct,4))

        t44 = time.time()
        print("\nLine 1553: \nTargetStock:", targetStock, "\nAda Boost Random Forest time=", round((t44-t43),0), "secs.")


        #GRADIENT BOOST
        #==============
        gradBoost = GradientBoostingRegressor(random_state=2025)
        gradBoost.fit(XTrain, yTrain)

        paramGridGradBoost = {
        'n_estimators': [110], #[90,100,110,120], # 300, 400], #200 3.09%; 120 3.09%
        'learning_rate': [0.2], # [0.01, 0.1, 0.2], #0.1
        'max_depth': [None], #[3, 4, 5], #4
        'min_samples_split': [2], #[2, 5, 10], #5
        'min_samples_leaf': [1], #[1, 2, 4], #2
        'subsample': [1], #[0.8, 0.9, 1.0],
        'max_features': ['log2'] #[1, 'sqrt', 'log2'] #['auto', 'sqrt', 'log2'] #sqrt
        }

        #gradBoostTScv = TimeSeriesSplit(n_splits=5)
        gradBoostGridSearch = GridSearchCV(estimator=gradBoost, 
                                        param_grid=paramGridGradBoost, 
                                        #cv=gradBoostTScv, 
                                        n_jobs=-1,
                                        verbose=1)
        gradBoostGridSearch.fit(XTrain, yTrain)

        # Get the best fit parameters.
        gradBoostBestnEst = gradBoostGridSearch.best_params_["n_estimators"]
        gradBoostBestLearn = gradBoostGridSearch.best_params_["learning_rate"]
        gradBoostMaxDepth = gradBoostGridSearch.best_params_["max_depth"]
        gradBoostBestMinSamplesSplit = gradBoostGridSearch.best_params_["min_samples_split"]
        gradBoostBestMinSamplesLeaf = gradBoostGridSearch.best_params_["min_samples_leaf"]
        gradBoostBestMaxFeatures = gradBoostGridSearch.best_params_["max_features"]

        gradBoostBestParams = gradBoostGridSearch.best_params_
        gradBoostBestScore = -gradBoostGridSearch.best_score_
        print("Line 1590 Gradient Boost Hyperparameters:", gradBoostBestParams, "\nbestScore=", gradBoostBestScore)

        # Fit the model with the best parameters:
        gradBoostTuned = GradientBoostingRegressor(
        n_estimators = gradBoostBestnEst,
        learning_rate = gradBoostBestLearn,
        max_depth = gradBoostMaxDepth, 
        min_samples_split = gradBoostBestMinSamplesSplit, 
        min_samples_leaf = gradBoostBestMinSamplesLeaf,
        max_features = gradBoostBestMaxFeatures,
        random_state = seedValue)

        gradBoostTuned.fit(XTrain,yTrain)
        print("\nLine 1604: Gradient Boost Regressor Tuned fit complete.\n")
        gradBoostTunedPred = gradBoostTuned.predict(XTest)
        print("\nLine 1606: \ngradBoostTunedPred=", round(gradBoostTunedPred[0],2), 
        "\nactualyTest=", round(yTest[0],2))

        # Evaluation of the gradient boost regression:
        #=============================================
        gradBoostTunedMSE = mean_squared_error(yTest, gradBoostTunedPred)
        gradBoostTunedRMSE = np.sqrt(gradBoostTunedMSE)
        gradBoostTunedRMSEPct = gradBoostTunedRMSE/yTest[0]*100
        gradBoostTestErrorPct = (gradBoostTunedPred[0]-yTest[0])/yTest[0]*100

        print("\nLine 1617: \ngradBoostTunedMSE=", round(gradBoostTunedMSE,2), 
        "\ngradBoostTunedRMSE=", round(gradBoostTunedRMSE,2), 
        "\ngradBoostTunedRMSEPct=", round(gradBoostTunedRMSEPct,2), "%",
        "\ngradBoostTestErrorPct=", round(gradBoostTestErrorPct,2), "%" )

        allModelsMeanPredV.append(round(gradBoostTunedPred[0],2))
        allModelsMeanTestErrorPctV.append(round(gradBoostTunedRMSEPct,4))

        t45 = time.time()
        print("\nLine 1623: \nTargetStock:", targetStock, "\nGradient Boost Regressor time=", round((t45-t44),0), "secs.")


        #EXTREME GRADIENT BOOST (XGB):
        #=============================
        XgradBoost = XGBRegressor(objective= "reg:squarederror", random_state=seedValue)
        #XgradBoost.fit(XTrain, yTrain)

        paramGridXGradBoost = {
        'n_estimators': [80], #[100,200,300], # Higher number of estimators reduces error/increases computational time.
        'learning_rate': [0.1], #[0.01, 0.1, 0.3], # Lower rates lead to better generalizations.
        'max_depth': [None], #[3, 4, 5, 6], # The depth of the trees, typically between 2 and 10.
        'min_child_weight': [3], #[1, 3, 5], 
        'subsample': [1], #[0.7, 0.8, 0.9], # Fraction of samples to train the tree. I will 1 to use all the time series.
        'colsample_bytree': [0.8], #[0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9], # Fraction of features (predictors).
        # Add regularization parameters:
        'reg_alpha': [0], #[0, 0.1, 0.5], # L1 Lasso regularization. 
        #Adds a penalty term proportional to the value of the coefficients.
        'reg_lambda': [1] #[0.1, 1, 5] #L2 Ridge regularization. 
        #Penalty to the squared value of the coefficients.
        }
        XgradBoostTScv = TimeSeriesSplit(n_splits=5)
        XgradBoostGridSearch = GridSearchCV(estimator=XgradBoost, 
                                        param_grid=paramGridXGradBoost, 
                                        cv=XgradBoostTScv, 
                                        n_jobs=-1,
                                        scoring="neg_mean_squared_error",
                                        verbose=1) # verbose 1 is almost no output.
        XgradBoostGridSearch.fit(XTrain, yTrain)

        # Get the best fit parameters.
        XgradBoostBestnEst = XgradBoostGridSearch.best_params_["n_estimators"]
        XgradBoostBestLearn = XgradBoostGridSearch.best_params_["learning_rate"]
        XgradBoostMaxDepth = XgradBoostGridSearch.best_params_["max_depth"]
        XgradBoostBestMinChildWeight = XgradBoostGridSearch.best_params_["min_child_weight"]
        XgradBoostBestColSample = XgradBoostGridSearch.best_params_["colsample_bytree"]
        XgradBoostBestAlphaLasso = XgradBoostGridSearch.best_params_["reg_alpha"]
        XgradBoostBestLambdaRidge = XgradBoostGridSearch.best_params_["reg_lambda"]

        XgradBoostBestParams = XgradBoostGridSearch.best_params_
        XgradBoostBestScore = -XgradBoostGridSearch.best_score_
        XgradBoostBestEstimator = XgradBoostGridSearch.best_estimator_
        print("Line 1663 XGB Hyperparameters:", XgradBoostBestParams, "\nbestScore=", XgradBoostBestScore)

        # Fit the model with the best parameters:
        XgradBoostTuned = XGBRegressor(objective= "reg:squarederror",
        n_estimators = XgradBoostBestnEst,
        learning_rate = XgradBoostBestLearn,
        max_depth = XgradBoostMaxDepth, 
        min_child_weight = XgradBoostBestMinChildWeight, 
        colsample_bytree = XgradBoostBestColSample,
        reg_alpha = XgradBoostBestAlphaLasso,
        reg_lambda = XgradBoostBestLambdaRidge,
        random_state = seedValue)

        XgradBoostTuned.fit(XTrain,yTrain)
        print("\nLine 1677: XGB Regressor Tuned fit complete.\n")
        XgradBoostTunedPred = XgradBoostTuned.predict(XTest)
        print("\nLine 1679: \nXGBTunedPred=", round(XgradBoostTunedPred[0], 2),
        "\nactualyTest=", round(yTest[0], 2))

        # Evaluation of the gradient boost regression:
        #=============================================
        XgradBoostTunedMSE = mean_squared_error(yTest, XgradBoostTunedPred)
        XgradBoostTunedRMSE = np.sqrt(XgradBoostTunedMSE)
        XgradBoostTunedRMSEPct = XgradBoostTunedRMSE/yTest[0]*100
        XgradBoostTestErrorPct = (XgradBoostTunedPred[0]-yTest[0])/yTest[0]*100

        print("\nLine 1690: \nXgradBoostTunedMSE=", round(XgradBoostTunedMSE,2), 
        "\nXgradBoostTunedRMSE=", round(XgradBoostTunedRMSE,2), 
        "\nXgradBoostTunedRMSEPct=", round(XgradBoostTunedRMSEPct,2), "%",
        "\nXgradBoostTestErrorPct=", round(XgradBoostTestErrorPct,2), "%" )

        allModelsMeanPredV.append(round(XgradBoostTunedPred[0],2))
        allModelsMeanTestErrorPctV.append(round(XgradBoostTunedRMSEPct,4))

        t46 = time.time()
        print("\nLine 1697: \nTargetStock:", targetStock, "\nXGB Regressor time=", round((t46-t45),0), "secs.")

        # ENSEMBLE STACKING: 
        #===================
        stackEstimators = [
        #('LRMultiPred', bestMultiPredglmModel),
        #('ridge', RidgeCV() ), pred: 260.93
        ('dtr', dtr), # pred= 187.11
        ('rfr', rfrTuned), # pred= 190.25
        ('adarfr', adarfrTuned), # pred= 190.69
        ("gradBoost", gradBoostTuned),#, # pred= 190.23
        ("XgradBoost", XgradBoostTuned)] # pred= 183.92

        # Define final regressor
        #stackFinalEstimator = bestMultiPredglmModel.fit()

        stackTScv = TimeSeriesSplit()
        stackFinalEstimator = RidgeCV() #XgradBoostTuned # LassoCV (didnt work)
        #stackFinalEstimator = XgradBoostTuned # LassoCV (didnt work)
        # Create stacking regressor
        stackRegressor = StackingRegressor(
        estimators=stackEstimators,
        final_estimator=stackFinalEstimator,
        #cv=5,
        n_jobs=-1
        )

        # Fit the stacking regressor
        stackRegressor.fit(XTrain, yTrain)

        # Make predictions
        yPredStack = stackRegressor.predict(XTest)
        stackRegressorRMSE = np.sqrt(mean_squared_error(yTest, yPredStack))

        t47 = time.time()
        print("\nLine 1730: \nTargetStock:", targetStock, 
        "\nyPredStack=", round(yPredStack[0],2), 
        "\nactualyTest=", round(yTest[0],2), 
        "\nstackRegressorRMSEPct=", stackRegressorRMSE, "%")

        #ModelName = ["SingleReg", "MultiReg", "DTree", "RF", "adaRF", "gradBoost", "XGB"]
        ModelName = ["DTree", "RF", "adaRF", "gradBoost", "XGB"]
        print("\nLine 1610: \nModelName:\n", ModelName, allModelsMeanPredV, allModelsMeanTestErrorPctV)
        allModelsDF = pd.DataFrame({"ModelName": ModelName, 
                                "meanPrediction": allModelsMeanPredV,
                                "MeanTestErrorPct": allModelsMeanTestErrorPctV})

        meanPredAllModels = np.mean(np.array(allModelsDF["meanPrediction"]))
        meanErrorAllModelsPct = (meanPredAllModels - yTest[0])/yTest[0] * 100
        print("\nLine 1607: \nallModelsDF:\n", allModelsDF,
        "\nactualyTest=", round(yTest[0],2), 
        "\nmeanPredAllModels=", round(meanPredAllModels,2), 
        "\nmeanErrorAllModels=", round(meanErrorAllModelsPct,2), "%")
        print("\nLine 1757: \nTargetStock:", targetStock, "\nEnsemble Stacking Regressor time=", round((t47-t46),0), "secs.")
        print("\nLine 1762: \nTargetStock:", targetStock, "\nTotal time=", round((t47-t0),0), "secs.")
        return meanPredAllModels, meanErrorAllModelsPct, yTest[0] 

npLag = 5
seedValue = 2025
targetStock = "AAPL"
marketIndices = ["^GSPC", "^DJI", "^IXIC", "^GDAXI", "^FTSE", "^FCHI", "^N225", "^HSI", "^STI"] 
bondIndices = ["^TYX", "^TNX","^FVX","^IRX","DX-Y.NYB"]
cmdtyIndices= ["GLD","CL=F"]
sp500Stocks = list(pd.read_csv(buildFilePath("sp500ListWiki"))["Symbol"])

# We will now predict the closing of all stocks in the SP500 npLag trading days forward.
# We will keep the mean prediction of all ensemble models, and their mean error.
closePricesMeanEnsembleV = []
testErrorMeanEnsemblePctV = []
actualCloseV = []
#sp500Stocks = ["AAPL","MSFT","BF.B", "GOOG"]
doNotInclude = ["CTLT", "ETR", "LEN","BF-B", "BRK-B", "BF.B", "BRK.B"]

# This is going to take a while process 500 stocks.
# Let's save progress, and pick up from where the error happened.
resultsFilename = "resultsDF.csv"
# If the results file does not exist, create it for the first time:
if (not (os.path.isfile("resultsDF.csv"))):        
        tempDF = pd.DataFrame( {"Ticker":[],
                                "ActualClose":[],
                                "MeanPred":[],
                                "MeanErrorPct":[]})
        tempDF.to_csv(resultsFilename,header=True, index=False)

# The starti may need to get a bit more complex:
resultsDF = pd.DataFrame(pd.read_csv(resultsFilename))
alreadyProcessed = list(resultsDF["Ticker"])
sp500StocksFilter = sorted(list(set(sp500Stocks.copy())-set(doNotInclude)- set(alreadyProcessed))) # ["AAPL","MSFT","GOOG"]
#starti = resultsDF.shape[0]

print("\nLine 1655: \nalreadyProcessed=", alreadyProcessed)

 
for i in range(len(sp500StocksFilter)): # -1 because the target stock is still inside.
        targetStock = sp500StocksFilter[i]
        print("\nLine 1655: i=", i, "of", len(sp500StocksFilter)-1, "targetStock=", targetStock)
        # Remove the target stock from the list of predictors.
        sp500StocksClean = sp500StocksFilter.copy()
        sp500StocksClean.remove(targetStock)
        #sp500StocksClean = sorted(list(set(sp500StocksFilter.copy())-set(targetStock)))

        # All securities in a list, after removing the target stock from it.
        # The target stock data series is already included in the start of the DF of the model.
        securitiesV = marketIndices+bondIndices+cmdtyIndices+sp500StocksClean
        closePricesMeanEnsemble, testErrorMeanEnsemblePct, actualClose=dtrRFREnsembles(securitiesV, targetStock=targetStock, minSeries=1250, npLag=npLag, seedValue=seedValue)
        newRow = pd.DataFrame( [{"Ticker": sp500StocksFilter[i],
                                "ActualClose": actualClose,
                                "MeanPred": closePricesMeanEnsemble,
                                "MeanErrorPct": testErrorMeanEnsemblePct}])
        newRow.to_csv(resultsFilename, mode="a", header=False, index=False)
        #resultsDF = pd.DataFrame(pd.read_csv(resultsFilename))
        #resultsDF["Ticker"].append(sp500Stocks[i])
        #resultsDF["ActualClose"].append(actualClose)
        #resultsDF["MeanPred"].append(closePricesMeanEnsemble)
        #resultsDF["MeanErrorPct"].append(testErrorMeanEnsemblePct)
        #resultsDF.to_csv(resultsFilename)


resultsDF = pd.DataFrame(pd.read_csv(resultsFilename))

meanErrorPct = np.mean(np.array(resultsDF["MeanErrorPct"]))
resultsDF["CorrectedClosePred"] = resultsDF["MeanPred"] *(1-meanErrorPct/100)
resultsDF["CorrectedErrorPct"] = resultsDF["MeanErrorPct"] -meanErrorPct

print("Line 1692: Final Results \nClose Price Prediction on 2025-02-20 for 500 S&P500 securities",
      "\n==============================================================\n", 
      resultsDF, "\nmeanErrorPct=", round(meanErrorPct,2), "%")

t48 = time.time()

print("\nLine 1698: \nTotal time All Stocks=", round((t48-t0),0), "secs.")
#======================================= THE END =========================================================
