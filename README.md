# House Price Prediction

## Project Description
House Price Prediction is an AI and Machine Learning project designed to predict house prices using available housing data.

## Project Files
- House_price1.csv - Dataset
- House_price_prediction2.py - Main Python file
- chart.py - Chart and visualization file

## Technologies Used
- Python
- Machine Learning
- Pandas
- NumPy
- AI Model
## How to Run

1. Install Python on your computer.
2. Install the required Python libraries.
3. Open the project folder in VS Code.
4. Run the following command:

   python House_price_prediction2.py

## Team Members
- Het Italiya
- Roll No: 529

- Avasar Amipara
- Roll No: 540


## Output

--------------------------------------------------
       Data set
--------------------------------------------------
1000 Data in this project
--------------------------------------------------
       Understanding Dataset
--------------------------------------------------

Shape : (1000, 10)

Columns : ['Area', 'BHK', 'Size_sqft', 'Bathrooms', 'Parking', 'Age_Years', 'Floor', 'Total_Floors', 'Furnished', 'Price_INR']

--------------------------------------------------
       Quantitative Data
--------------------------------------------------

quantitative data:int64 and float64
quantitative data:      BHK  Size_sqft  Bathrooms  Parking  Age_Years  Floor  Total_Floors  Price_INR
0      2       1009          2        1         19      9            11    4334906
1      1        694          1        1          1      5            16    3153605
2      4       1991          4        1          7      7             8   10109899
3      2       1047          2        1         15      4            12    4084870
4      2        969          2        1          6      9            17    6216619
..   ...        ...        ...      ...        ...    ...           ...        ...
995    2       1096          2        1         15      3            12    2794800
996    3       1528          3        1          9      5             9    5956816
997    1        534          1        1          7      5            16    2634072
998    3       1499          3        1          9      8            11    8184540
999    2       1052          2        1         11      5             5    4775028

[1000 rows x 8 columns]
quantitative column labels:
 Index(['BHK', 'Size_sqft', 'Bathrooms', 'Parking', 'Age_Years', 'Floor',
       'Total_Floors', 'Price_INR'],
      dtype='str')

--------------------------------------------------
       Qualitative Data
--------------------------------------------------

 qualitative data:object and category
e:\House_price_prediction\House_price_prediction2.py:37: Pandas4Warning: For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated and will be removed in a future version. Explicitly pass 'str' to `include` to select them, or to `exclude` to remove them and silence this warning.
See https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#string-migration-select-dtypes for details on how to write code that works with pandas 2 and 3.
  qualitative=df.select_dtypes(include=['object','category'])
Qualitative data:              Area    Furnished
0             Pal         Semi
1        Palanpur         Semi
2             Pal        Fully
3        Palanpur         Semi
4         Bhimrad        Fully
..            ...          ...
995       Dindoli  Unfurnished
996  Jahangirpura         Semi
997           Pal         Semi
998        Althan  Unfurnished
999        Adajan         Semi

[1000 rows x 2 columns]
qualitative column labels:
 Index(['Area', 'Furnished'], dtype='str')

--------------------------------------------------
       EDA
--------------------------------------------------
       Area  BHK  Size_sqft  Bathrooms  Parking  Age_Years  Floor  Total_Floors Furnished  Price_INR
0       Pal    2       1009          2        1         19      9            11      Semi    4334906
1  Palanpur    1        694          1        1          1      5            16      Semi    3153605
2       Pal    4       1991          4        1          7      7             8     Fully   10109899
3  Palanpur    2       1047          2        1         15      4            12      Semi    4084870
4   Bhimrad    2        969          2        1          6      9            17     Fully    6216619
             Area  BHK  Size_sqft  Bathrooms  Parking  Age_Years  Floor  Total_Floors    Furnished  Price_INR
995       Dindoli    2       1096          2        1         15      3            12  Unfurnished    2794800
996  Jahangirpura    3       1528          3        1          9      5             9         Semi    5956816
997           Pal    1        534          1        1          7      5            16         Semi    2634072
998        Althan    3       1499          3        1          9      8            11  Unfurnished    8184540
999        Adajan    2       1052          2        1         11      5             5         Semi    4775028
Area            0
BHK             0
Size_sqft       0
Bathrooms       0
Parking         0
Age_Years       0
Floor           0
Total_Floors    0
Furnished       0
Price_INR       0
dtype: int64
               BHK    Size_sqft    Bathrooms      Parking    Age_Years        Floor  Total_Floors     Price_INR
count  1000.000000  1000.000000  1000.000000  1000.000000  1000.000000  1000.000000   1000.000000  1.000000e+03
mean      2.511000  1278.326000     2.511000     1.260000     9.928000     6.772000     12.629000  6.984928e+06
std       0.941682   427.236486     0.941682     0.438854     5.980517     4.427824      4.524802  3.389567e+06
min       1.000000   504.000000     1.000000     1.000000     0.000000     1.000000      5.000000  1.500000e+06
25%       2.000000  1005.750000     2.000000     1.000000     5.000000     3.000000      9.000000  4.398334e+06
50%       3.000000  1402.000000     3.000000     1.000000    10.000000     6.000000     13.000000  6.382040e+06
75%       3.000000  1548.250000     3.000000     2.000000    15.000000     9.250000     17.000000  8.946834e+06
max       4.000000  2050.000000     4.000000     2.000000    20.000000    20.000000     20.000000  1.962920e+07
Area             10
BHK               4
Size_sqft       549
Bathrooms         4
Parking           2
Age_Years        21
Floor            20
Total_Floors     16
Furnished         3
Price_INR       995
dtype: int64

--------------------------------------------------
       Correlation and covariation
--------------------------------------------------
                       BHK     Size_sqft     Bathrooms        Parking     Age_Years          Floor   Total_Floors     Price_INR
BHK           8.867658e-01  3.985500e+02  8.867658e-01       0.221361 -3.525606e-01      -0.082575       0.077659  2.222051e+06
Size_sqft     3.985500e+02  1.825310e+05  3.985500e+02      99.904144 -1.697503e+02     -40.489161      30.676623  1.018773e+09
Bathrooms     8.867658e-01  3.985500e+02  8.867658e-01       0.221361 -3.525606e-01      -0.082575       0.077659  2.222051e+06
Parking       2.213614e-01  9.990414e+01  2.213614e-01       0.192593 -1.003804e-01      -0.061782       0.038498  5.933185e+05
Age_Years    -3.525606e-01 -1.697503e+02 -3.525606e-01      -0.100380  3.576658e+01      -1.263680      -0.431143 -3.872882e+06
Floor        -8.257457e-02 -4.048916e+01 -8.257457e-02      -0.061782 -1.263680e+00      19.605622      10.070482  1.324112e+05
Total_Floors  7.765866e-02  3.067662e+01  7.765866e-02       0.038498 -4.311431e-01      10.070482      20.473833  3.187674e+05
Price_INR     2.222051e+06  1.018773e+09  2.222051e+06  593318.510110 -3.872882e+06  132411.207127  318767.352042  1.148916e+13
                   BHK  Size_sqft  Bathrooms   Parking  Age_Years     Floor  Total_Floors  Price_INR
BHK           1.000000   0.990627   1.000000  0.535646  -0.062602 -0.019804      0.018226   0.696154
Size_sqft     0.990627   1.000000   0.990627  0.532838  -0.066436 -0.021403      0.015869   0.703501
Bathrooms     1.000000   0.990627   1.000000  0.535646  -0.062602 -0.019804      0.018226   0.696154
Parking       0.535646   0.532838   0.535646  1.000000  -0.038246 -0.031794      0.019388   0.398863
Age_Years    -0.062602  -0.066436  -0.062602 -0.038246   1.000000 -0.047721     -0.015932  -0.191052
Floor        -0.019804  -0.021403  -0.019804 -0.031794  -0.047721  1.000000      0.502644   0.008822
Total_Floors  0.018226   0.015869   0.018226  0.019388  -0.015932  0.502644      1.000000   0.020784
Price_INR     0.696154   0.703501   0.696154  0.398863  -0.191052  0.008822      0.020784   1.000000

--------------------------------------------------
       Value Counts
--------------------------------------------------
Area
Pal             109
Piplod          109
Palanpur        105
Adajan          105
Jahangirpura    102
Dindoli         102
Vesu            102
Bhimrad          93
City Light       89
Althan           84
Name: count, dtype: int64
BHK
2    342
3    340
4    163
1    155
Name: count, dtype: int64
Size_sqft
1122    6
1417    6
985     6
1139    6
1533    5
       ..
595     1
996     1
504     1
1937    1
534     1
Name: count, Length: 549, dtype: int64
Bathrooms
2    342
3    340
4    163
1    155
Name: count, dtype: int64
Parking
1    740
2    260
Name: count, dtype: int64
Age_Years
2     60
7     59
10    56
18    54
13    53
16    52
4     52
15    51
6     50
12    50
1     48
17    48
5     48
19    47
11    47
3     44
9     43
0     38
14    38
20    34
8     28
Name: count, dtype: int64
Floor
5     109
3      96
1      93
4      85
2      84
6      83
7      82
8      67
9      51
10     51
11     42
12     34
13     27
14     27
16     18
17     16
15     14
18     10
19      8
20      3
Name: count, dtype: int64
Total_Floors
8     72
17    71
15    70
13    69
16    65
19    65
12    64
11    63
9     62
18    60
6     58
14    58
20    57
10    57
7     55
5     54
Name: count, dtype: int64
Furnished
Fully          363
Semi           320
Unfurnished    317
Name: count, dtype: int64
Price_INR
1500000    3
9514050    2
3213000    2
4465125    2
4334906    1
          ..
2794800    1
5956816    1
2634072    1
8184540    1
4775028    1
Name: count, Length: 995, dtype: int64

--------------------------------------------------
       Data preprocessing
--------------------------------------------------

--------------------------------------------------
       categorical and numerical columns
--------------------------------------------------

--------------------------------------------------
       Regression
--------------------------------------------------

--------------------------------------------------
      R2 and RSS calculation
--------------------------------------------------
r2 value: 0.9521696170824505
RSS: 107098499434109.44

--------------------------------------------------
       MAE and MSE calculation
--------------------------------------------------
mean mae: 543936.8345919604
mean mse: 535492497170.5472

--------------------------------------------------
       RMSE calculation
--------------------------------------------------
RMSE: 731773.5286074149
Enter Area: vesu
Enter BHK: 4
Enter Size in sqft: 2400
Enter Bathrooms: 4
Enter Parking: 3
Enter Age in Years: 5
Enter Floor: 4
Enter Total Floors: 13
Enter Furnished Type: 3

--------------------------------------------------
             PREDICTION RESULT
--------------------------------------------------
Area              : vesu
BHK               : 4
Size              : 2400.0 sqft
Bathrooms         : 4
Parking           : 3
Age               : 5 years
Floor             : 4
Total Floors      : 13
Furnished Type    : 3

Predicted House Price : ₹ 13,701,092.81

==================================================
          PREDICTION COMPLETED
==================================================

