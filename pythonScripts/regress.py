"""
A file to linearly downscale predictors

Right now this just does linear regression and uses all predictors that are passed.
Updated 7.16.2020, K. Wheelan

Usage: regress.py <lat> <lon> <obs filepath> <location to save data> <any pred file paths>+

 """

#TODO:
#add a switch for Lasso
#add stochasticity
#add a switch for two step regression (precip)

#import necessary packages
from regression_methods import *
import_dependencies()

#Set globals
monthsAbrev = ['Jan','Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthsFull = ['January','February', 'March','April','May','June','July','August','September','October','November','December']

#set variables based on commandline arguments
if len(sys.argv) < 5:
    exit("Usage: regress.py <lat> <lon> <obsPath> <save_path> <preds>+")

lat = float(sys.argv[1])
lon = float(sys.argv[2])
obsPath = sys.argv[3] #filepath for obs data
save_path = sys.argv[4]
preds = [ path for path in sys.argv[5:] ] #paths for preds files

print("Progress:")
print(f"Lat: {lat}, Lon: {lon}")

#import predictors
predictors = load_predictors()
print("Loaded predictor files")


#standardize data, trim dates, add month and constant cols
predictors = standardize(predictors)
X_all, Y_all = prep_data(obsPath, predictors)
X_all, Y_all, all_preds = add_month(X_all, Y_all)
X_all, all_preds = add_constant_col(X_all, all_preds)
print("Loaded obs data.")

#separate testing and training data by even and odd years
X_train, X_test = evenOdd(X_all)
Y_train, Y_test = evenOdd(Y_all)

#creating a month filter
for data in [X_train, X_test, Y_train, Y_test]:
    data = add_month_filter(data)

print("Prepped data for regression")

## TODO:
#saving training/testing data
# print("Saved prepped data")

# Fit regression model
#TODO verify this chunk
preds_to_drop = ["month", "lat", "lon"]
preds_to_keep = [x for x in all_preds if not x in preds_to_drop]

#fit a different model for each month
coefMatrix = fit_monthly_linear_models(X_train, Y_train, preds_to_keep)
print("Fit linear model.")

#get linear hand-selected predictor test data
#x_test_subset = [np.matrix([X_test.sel(time = month)[key].values for key in preds_to_keep]).transpose() for month in range(1,13)]


#saves the betas
save_betas(file_path, coefMatrix)
print("Saved betas.")

#predict for all data using betas
final_predictions = predict_linear(X_all, coefMatrix)
print("Calculated predictions for testing and training data.")

# TODO: add stochasticity, transformations

save_preds(save_path, final_predictions)
print("Saved predictions.")
print("Done.")
