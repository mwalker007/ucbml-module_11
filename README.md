### Business Understanding

#### Business Objective

The primary business objective is to provide clear recommendations our client, a used car dealership, on what are the key criteria that affect what a buyer is willing to pay for a used car. 

#### Data Mining Goals

Given the dataset of 426K vehicle purchases which includes the features (id, region, year, manufacturer, model, condition, cylinders,
fuel, odometer, title_status, transmission, VIN, drive, size, type, paint_color, state) along with the target selling price, determine which features are most important in determining the price paid for a used vehicle. The desired output is a list of the most important features in rank order.  

#### Project Plan

1) Aquire and explore the dataset for understanding
2) Prepare the data including cleaning and preprocessing
3) Try various modeling techniques in order to find the best model fit for the data and feature importance
4) Evaluate model outputs and repeat steps 3 & 4 as needed
5) Document analysis findings

### Data Understanding

#### Collect Initial Data
The provided dataset contains sale price and vehicle details on 426K cars 

#### Data Description

##### Dataset contains 426,880 rows and includes the following 18 columns: 
id, region, price, year, manufacturer, model, condition, cylinders, fuel,
odometer, title_status, transmission, VIN, drive, size, type, paint_color, state
 
##### Identify Numeric and Categorical Columns
Numerical Columns: ['id', 'price', 'year', 'odometer']
Categorical Columns: ['region', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'VIN', 'drive', 'size', 'type', 'paint_color', 'state']

##### Explored Relationship of Numerical Columns
* Determined that the numerical columns are not highly correlated

##### Explored Counts of the Categorical Numerical Columns through histograms
See Notebook for Chart Output

### Data Preparation

##### Data Selection

##### Drop the following columns:
*  id and VIN because they are not values people make buying decisions on unless in the case of the VIN the vehicle is extremely unique
*  region because there is only a relatively small amount of data for each region

##### Handle the remaining columns as follows:
* Price - drop values where price > 80,000 and < 1000 since typically represents special circumstances like not-running or high end vehicles 
* Year - only include vehicle year 2000 or newer since vehicles >20 years old have different considerations than modern ones
* Manufacturer - only include with 5000 or more sales in the dataset since since rare/ucommon vehicles are priced based on different considerations
* Model - only include with 500 or more sales in the dataset since rare/uncommon vehicles are priced based on different considerations
* Condition - A) Drop new and salvage since not enough examples in the dataset B) create an order - fair, good, excellent, like new, new 
* Cylinders - A) Keep 4,6,8 and drop the rest since they are uncommon in the dataset and real life B) create an order - 4,6,8
* Fuel - Keep only Gas since other options are not well represented in the dataset
* Odometer - Keep only 200,000 or below since vehicles above that are not common and typically are valued differently
* Title Status - Keep only Clean since there is limited data for the other values 
* Transmission - Keep only Automatic since there is limited data for the other values 
* drive - Keep all
* size - Create an order - sub-compact,compact,mid-size,full-size
* type - Keep all except off-road and bus becuase they are uncommon and not represented well in the dataset
* paint - Keep paint colors with above 1,000 samples in the dataset since they are the most common
* State - Don't include in modeling since many states are not well represented in the dataset

##### Data Cleansing
* A review of remaining values after initial selection steps showed that there are not enough rows with size values relative to the rest of the columns to be useful so it is dropped
* Dropped any remaining rows with NaN values to better support model training algorithms

###### Review Remaining Data
* Data columns (total 14 columns):
* Column        Non-Null Count  Dtype   
*  0   price         25175 non-null  int64  
*  1   year          25175 non-null  float64
*  2   manufacturer  25175 non-null  object 
*  3   model         25175 non-null  object 
*  4   condition     25175 non-null  object 
*  5   cylinders     25175 non-null  object 
*  6   fuel          25175 non-null  object 
*  7   odometer      25175 non-null  float64
*  8   title_status  25175 non-null  object 
*  9   transmission  25175 non-null  object 
*  10  drive         25175 non-null  object 
*  11  type          25175 non-null  object 
*  12  paint_color   25175 non-null  object 
*  13  state         25175 non-null  object 
 
### Modeling
##### Explore the realtionship between Year and Price
1. As expected more recent vehicles typically cost more but there are exceptions

##### Explore the realtionship between Odometer and Price
1. There is an inverse relationship between Odometer and Price with some outliers

##### Explore the realtionship between State and Price
1. There is not a clear relationship between state and price 

##### Data Selection
1.  Since fuel,title_status, transmission have all been limited to one value, they can be dropped for modeling purposes. State is also dropped since there isn't a strong correlation between state and price.

##### Training
1. Trained a Linear Regression model using year and odometer values and determine best order for Polynomials
    1. Train MSEs:['57668136.03', '53651593.50', '53514841.07', '53146321.38', '52826628.64']
    2. Test MSEs:['58417291.92', '54526704.52', '54413949.87', '53949332.01', '53601885.08']
    3. Order 2 had the nearly the best Test MSE so degree=2 will be used going forward inorder to minimize overfitting
##
2. Trained a second order LinReg model for year and odometer along with determining coefficients/feature importance
    1. Train MSE:  53,651,593.50
    2. Test MSE:  54,526,704.52
    3. feature coef
        1. odometer  1.351096e+04
        2. year  -1.427082e+06
        3. odometer^2  1.732982e+03
        4. odometer year  -1.720482e+04
        5. year^2  1.432260e+06
##
3. Trained a Ridge Regression model using only first order year and odometer values to find the best Alpha and see if Ridge has a lower MSE than Linreg
    1. Test MSE: 58,417,286.93
    2. Best Alpha: 10.0
    3. This was not better than a 2nd order polynomial LinReg model
##
4. Trained a Ridge Regression model using second order year and odometer values to find the best Alpha and if Ridge has a lower MSE than LinReg
    1. Test MSE: 54,526,035.19
    2. Best Alpha: 10.0
    3. MSE was similar to LinReg when Polynomial degree=2. Both were ~54.5M
##
5. Trained a Lasso model using 1st order year and odometer values to determine feature importance
    1. Train MSE:  57,668,137.18
    2. Test MSE:  58,417,325.01
    3. Lasso Coefficients:
        1. feature     coef
        2. year     4673.874150
        3. odometer     -2792.832738
##
6. Trained a Lasso model using 2nd order year and odometer values and examine feature importance
    1. Train MSE:  56510282.66
    2. Test MSE:  57173256.88
    3. Lasso Coefficients:
        1. Feature     coef
        2. year     7313.202555
        3. odometer     -5194.636102
        4. Year^2     -2614.098847
        5. year odometer     -1648.448340
        6. Odometer^2     4201.894958
##
7. Trained a Lasso model using 1st order year and odometer values along with ordinal encoded cyclinders and condition values to determine feature importance
    1. Train MSE:  33,943,901.03
    2. Test MSE:  34,767,370.60
        1. feature   coef
        2. year    574.933204
        3. odometer     4894.743268
        4. cylinders    5139.360788
        5. condition    -2989.811602
    3. With more features, the model had a much lower MSE and better fit for the test dataset. 
    4. RESULTS: When trained on only year, odometer, cylinders, and condition, the most important feature was cylinders followed by odometer,condition and year
##
8. Created LinReg models for the numerical features + one of the categorical feature at a time and then compare MSEs
    1.    feature      test mse
    2.    manufacturer  5.790257e+07
    3.    model         5.790257e+07
    4.    drive         2.946073e+07
    5.    type          2.888626e+07
    6.    paint_color   3.248619e+07
    7.    RESULTS: When combined with: [odometer,cylinders,condition,year],type was the most important feature followed by drive,paint_color, model and manufacturer.
##
9. Created LinReg model for each of the  features trained one at a time and then compare MSEs
    1.    feature      test mse
    2.    odometer  7.380323e+07
    3.    cylinders  8.856595e+07
    4.    condition  9.490218e+07
    5.    year  5.753197e+07
    6.    manufacturer  9.133293e+07
    7.    model  6.850711e+07
    8.    drive  8.592261e+07
    9.    type  8.106760e+07
    10.   paint_color  1.004205e+08 
    11.   RESULTS: When modeled individually,the order of feature importance was year,model,odometer,type,drive,cylinders,manufacuturer, condition and paint color.
##
10. Similar to above above where I created a LinReg model for each of the  features trained one at a time and then compare MSEs, but now varying the degree of the Polynomial Transformer
    1. RESULTS: Regardless of Polynomial order between 1-5, when modeled individually,the order of feature importance was still year,model,odometer,type,drive,cylinders,manufacuturer, condition and paint color.
##
11. Created Ridge model with Alpah=10 for the numerical features + one of the categorical feature at a time and then compare MSEs
    1. RESULTS: Similar to the LinReg model where the order of feature importance was year,model,odometer,type,drive,cylinders,manufacuturer,condition and paint color.

##
### Evaluation
#### Summary of Results
1. Multiple Linear Regression, Ridge, and Lasso were trained on a subset of the initial data and regardless of the model type the rank of feature importance was
    1. year
    2. model
    3. odometer
    4. type
    5. drive
    6. cylinders
    7. manufacuturer
    8. condition
    10. paint color.
2. This lines up with general expectations on what buyers typically prioritize when purchasing a vehicle

##### Possible Next Steps
1. There was considerable missing data in the dataset so it could be useful to impute the missing values vs. just deleting the rows that were missing any data
2. State was not factored into the modeling since there was a very uneven distribution of data. One potential way to address this would be to bucket the state values into 
US subregions such as North East, Mid-West, etc. to understand how price varies by selling location.
3. Explore other feature importance methods such as permutation_importance
4. As usual, having more data, especially data that helps have more records from the less common models, types, etc. would likely be usefull in getting more accurate predictions.

#### The Jupyter Notebook used to analyze the data can be found at [prompt_II.ipynb](./prompt_II.ipynb)
