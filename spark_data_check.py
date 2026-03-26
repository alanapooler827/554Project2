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
        lower = None,
        upper = None
    ):
        
        # check that upper and/or lower bounds are provided
        if lower is None and upper is None:
            print("Error: Must provide at least one of 'upper' or 'lower'")
            return self
        
        # check that column type is numeric
        col_type = dict(self.df.dtypes)[column]
        
        # if not numeric, print message and return unmodified df
        if col_type not in ('float', 'int', 'longint', 'bigint', 'double', 'integer'):
            print('Error: Column must of type float, int, longint, bigint, double, or integer')
            return self
        
        # check if column is within provided range
        if lower is not None and upper is not None:
            chk = F.col(column).between(lower, upper)
        elif lower is not None:
            chk = F.col(column) >= lower
        else:
            chk = F.col(column) <= upper

        # add new column to dataframe
        # if provided column is null, return null
        self.df = self.df.withColumn(
            'within_num_range',
            F.when(F.col(column).isNull(), F.lit(None))
            .otherwise(chk)
        )
        
        # return new data frame
        return self
    
    # method to check values of a string column
    def check_string_col(self, column, levels):

        # check that column is string type
        col_type = dict(self.df.dtypes)[column]
        if col_type != 'string':
            print('Error: Column type must be string.')
            return self

        # add new column to dataframe
        # if column is null, return null
        self.df = self.df.withColumn(
            'within_level',
            F.when(F.col(column).isNull(), F.lit(None))
            .otherwise(F.col(column).isin(levels))
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
    
    def summarize_min_max(self, column = None, group_col = None):
        
        # get column types from data frame
        type_dict = dict(self.df.dtypes)
        # define numeric column types
        numeric_types = ('float', 'int', 'longint', 'bigint', 'double', 'integer')
        
        # check if numeric column is provided
        if column is not None:
            
            # if not numeric, print message and return unmodified df
            if type_dict.get(column) not in numeric_types:
                print('Error: Column must of type float, int, longint, bigint, double, or integer')
                return self
            
            # if numeric column is provided, find min and max
            if group_col is None:
                # use .alias() to name new columns according to provided column
                result_df = self.df.agg(
                    F.min(column).alias(f"{column}_min"), 
                    F.max(column).alias(f"{column}_max")
                )
            
            # if numeric column is provided, find min and max with grouping column
            else:
                result_df = self.df.groupBy(group_col).agg(
                    F.min(column).alias(f"{column}_max"),
                    F.max(column).alias(f"{column}_max")
                )
            
            # return result as pandas dataframe
            return result_df.toPandas()
        
        # if column is not provided, summarize all numeric columns
        numeric_cols = [
            col_name for col_name in self.df.columns if type_dict.get(col_name) in numeric_types
        ]
        
        # find min and max for all numeric columns without grouping column
        if group_col is None:
            agg_exprs = []
            for col_name in numeric_cols:
                agg_exprs.extend([
                    F.min(F.col(col_name)).alias(f"{col_name}_min"),
                    F.max(F.col(col_name)).alias(f"{col_name}_max")
                ])

            return self.df.agg(*agg_exprs).toPandas()
    
        # find min and max for all numeric columns with grouping column
        df_list = []
        for col_name in numeric_cols:
            min_max_df = (
                self.df.groupBy(group_col)
                .agg(
                    F.min(F.col(col_name)).alias(f"{col_name}_min"),
                    F.max(F.col(col_name)).alias(f"{col_name}_max")
                )
                .toPandas()
            )
            df_list.append(min_max_df)
        
        # reduce result into one data frame
        result_df = reduce(lambda left, right: pd.merge(left, right, on=group_col), df_list)
        return result_df