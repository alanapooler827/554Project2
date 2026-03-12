# import libraries
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from functools import reduce
from pyspark.sql.types import *
import pandas as pd

# define class
class SparkDataCheck:

    # initialize the class
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # define class method to create instance while reading in csv file
    @classmethod
    def read_csv(cls, spark, file_path):

        # read in csv file
        df = spark.read.load(
            file_path,
            format = 'csv',
            inferSchema = 'true',
            header = 'true'
        )

        # 
        return cls(df)
    
    # define class method to create instance from pandas df
    @classmethod
    def read_pandas(cls, spark, pandas_df: pd.DataFrame):
        
        # create spark SQL data frame
        df = spark.createDataFrame(pandas_df)

        # return new instance of the class
        return cls(df)
    
    # method to check range of a numeric column
    def check_numeric_col(
        self,
        column,
        lower,
        upper
    ):
        
        # check that upper and/or lower bounds are provided
        if lower is None and upper is None:
            print("Error: Must provide at least one of 'upper' or 'lower'")
        
        # check that column type is numeric
        col_type = dict(self.df.dtypes)['column']
        # if not numeric, print message and return unmodified df
        if col_type not in ('float', 'int', 'longint', 'bigint', 'double', 'integer'):
            print('Error: Column must of type float, int, longint, bigint, double, or integer')
            return self
        
        # turn provided column into pyspark column object
        chk_col = F.col(column)
        
        # check if column is within provided range
        if lower is not None and upper is not None:
            chk = chk_col.between(lower, upper)
        elif lower is not None:
            chk = chk_col >= lower
        else:
            chk = chk_col <= upper

        # add new column to dataframe
        # if provided column is null, return null
        self.df = self.df.withColumn(
            'within_num_range',
            F.when(chk_col.isNull(), F.lit(None))
            .otherwise(chk_col)
        )
        
        # return new data frame
        return self
    
    # method to check values of a string column
    def check_string_col(self, column, levels):

        # check that column is string type
        col_type = dict(self.df.dtypes)['column']
        if col_type != 'string':
            print('Error: Column type must be string.')
            return self
        
        # turn provided column into pyspark column object
        chk_col = F.col(column)

        # add new column to dataframe
        # if column is null, return null
        self.df = self.df.withColumn(
            'within_level',
            F.when(chk_col.isNull(), F.lit(None))
            .otherwise(chk_col.isin(levels))
        )

        return self
    
    # method to check a column for missing values
    def check_missing_values(self, column):

        # add new column that checks whether value is null
        self.df = self.df.withColumn(
            'is_missing',
            F.col(column).isNull()
        )

        return self