from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array_contains
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Song Emotion Analysis") \
    .getOrCreate()

# Define the schema for the JSON data
schema = StructType([
    StructField("name", StringType(), True),
    StructField("artist", StringType(), True),
    StructField("lyric", StringType(), True),
    StructField("genre", StringType(), True),
    StructField("top_emotions", ArrayType(StringType()), True)
])

# Function to filter songs based on emotions
def filter_songs_by_emotions(df, emotions):
    # Perform filtering based on emotions
    filtered_df = df.filter(
        array_contains(col("top_emotions"), emotions[0]) |
        array_contains(col("top_emotions"), emotions[1]) |
        array_contains(col("top_emotions"), emotions[2])
    )
    return filtered_df

# Load the JSON data into a DataFrame with the defined schema
df = spark.read.json("data/test_songs_with_emotions.json", schema=schema)

# Show the DataFrame schema
# df.printSchema()

# Emotions to filter
emotions_to_filter = ["love", "surprise", "joy"]

# Apply the filter
filtered_df = filter_songs_by_emotions(df, emotions_to_filter)

# Show the filtered DataFrame
filtered_df.show()
