from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, to_date, hour, sha2, concat_ws, current_timestamp, coalesce,
    from_unixtime, to_timestamp, count, substring, avg, from_json, window,
    last, max, when
)
from pyspark.sql.types import *
import sys

# 1. KONFIGURACJA SPARKA
spark = (
    SparkSession.builder
        .appName("silver_merge")
        .enableHiveSupport()
        .config("spark.sql.warehouse.dir", "hdfs://node1/user/hive/warehouse")
        .config("spark.sql.catalogImplementation", "hive")
        .config("spark.sql.shuffle.partitions", "5") 
        .getOrCreate()
)

spark.conf.set("spark.sql.parquet.enableVectorizedReader", "false")
spark.conf.set("spark.sql.parquet.enableVectorizedWriter", "false")
spark.conf.set("spark.sql.parquet.outputTimestampType", "INT96")
spark.conf.set("spark.sql.legacy.parquet.int96RebaseModeInWrite", "CORRECTED")
spark.conf.set("spark.sql.legacy.parquet.datetimeRebaseModeInWrite", "CORRECTED")
spark.conf.set("spark.sql.parquet.writeLegacyFormat", "false")
spark.conf.set("spark.sql.columnVector.batchSize", "1024")
spark.conf.set("spark.sql.streaming.schemaInference", "true")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

# 2. SCHEMATY DANYCH

next_bike_schema = StructType([
    StructField("station_id", StringType()),
    StructField("last_reported", LongType()),
    
    StructField("num_bikes_available", IntegerType()),
    StructField("num_docks_available", IntegerType()),
    StructField("is_installed", IntegerType()),
    StructField("is_renting", IntegerType()),
    StructField("is_returning", IntegerType()),
    
    StructField("name", StringType()),
    StructField("short_name", StringType()),
    StructField("lat", DoubleType()),
    StructField("lon", DoubleType()),
    StructField("region_id", StringType()),
    StructField("capacity", IntegerType()),
    
    StructField("bike_id", StringType())
])

next_bike_csv_schema = next_bike_schema

next_bike_csv_schema = next_bike_schema

weather_schema = StructType([
    StructField("lon", DoubleType()),
    StructField("lat", DoubleType()),
    StructField("main_weather", StringType()),
    StructField("temp", DoubleType()),
    StructField("temp_feels_like", DoubleType()),
    StructField("temp_min", DoubleType()),
    StructField("temp_max", DoubleType()),
    StructField("wind_speed", DoubleType()),
    StructField("wind_deg", IntegerType()),
    StructField("datetime", LongType()),
    StructField("rain_1h", DoubleType())
])

bvg_schema = StructType([
    StructField("id", StringType()),
    StructField("origin", StringType()),
    StructField("destination", StringType()),
    StructField("departure", StringType()),
    StructField("arrival", StringType())
])

# 3. FUNKCJE NORMALIZUJĄCE

def normalize_nextbike(df):

    df = df.filter(col("station_id").isNotNull())

    df = df.withColumn("station_id", substring(col("station_id").cast("string"), 1, 50)) \
           .withColumn("ingestion_ts", current_timestamp())

    df = df.withWatermark("ingestion_ts", "10 minutes")

    df_merged = df.groupBy(
        window(col("ingestion_ts"), "5 minutes"), 
        col("station_id")
    ).agg(
        last("name", ignorenulls=True).alias("station_name_raw"),
        last("lat", ignorenulls=True).alias("lat"),
        last("lon", ignorenulls=True).alias("lon"),
        last("region_id", ignorenulls=True).alias("region_id"),
        
        max("num_bikes_available").alias("num_bikes_available"),
        max("num_docks_available").alias("num_docks_available"),
        max("capacity").alias("capacity"),
        
        max("ingestion_ts").alias("events_ts")
    )

    df = (
        df_merged
        .withColumn("dt", to_date("events_ts"))
        .withColumn("hour", hour("events_ts"))
        .withColumn("source", lit("nextbike"))
        .withColumn("station_name", coalesce(col("station_name_raw"), col("station_id")))
        .withColumn("area_id", coalesce(col("region_id"), lit("unknown_area")))
        .withColumn("bikes_available", coalesce(col("num_bikes_available"), lit(0)).cast(IntegerType()))
        .withColumn(
            "slots_available",
            coalesce(col("num_docks_available"), col("capacity"), lit(0)).cast(IntegerType())
        )
        .withColumn("is_reserved", lit(0))
        .withColumn("is_disabled", lit(0))
    )

    hash_cols = [
        "station_id", "station_name", "bikes_available", "slots_available", "events_ts"
    ]

    df = df.withColumn(
        "payload_hash",
        sha2(concat_ws("||", *[coalesce(col(c).cast("string"), lit("")) for c in hash_cols]), 256)
    )

    return df.select(
        "station_id", "station_name", "area_id", "lat", "lon",
        "bikes_available", "slots_available", "is_reserved", "is_disabled",
        "events_ts", "dt", "hour", "source", "payload_hash"
    )
def normalize_weather(df):
    df = df.withColumn("events_ts", from_unixtime(col("datetime").cast("long")).cast("timestamp")) \
           .withColumn("dt", to_date("events_ts")) \
           .withColumn("hour", hour("events_ts")) \
           .withColumn("ingestion_ts", current_timestamp()) \
           .withColumn("source", lit("weather"))

    df = df.filter(col("lon").isNotNull() & col("lat").isNotNull() & col("events_ts").isNotNull())

    df = df.withColumn("area_id", concat_ws("_", col("lon"), col("lat"))) \
           .withColumnRenamed("temp", "temperature_avg") \
           .withColumnRenamed("wind_speed", "wind_speed_avg") \
           .withColumnRenamed("rain_1h", "precipitation_sum")

    hash_cols = ["area_id", "temperature_avg", "wind_speed_avg", "precipitation_sum", "events_ts"]
    df = df.withColumn("payload_hash", sha2(concat_ws("||", *[col(c).cast("string") for c in hash_cols]), 256)) \
           .dropDuplicates(["source", "payload_hash"])

    return df.select(
        "area_id", "events_ts", "temperature_avg", "wind_speed_avg", "precipitation_sum",
        "dt", "hour", "ingestion_ts", "source", "payload_hash"
    )

def normalize_bvg(df):
    df = (
        df.withColumn("id", substring(col("id"), 1, 50))
          .withColumn("origin", substring(col("origin"), 1, 255))
          .withColumn("destination", substring(col("destination"), 1, 255))
    )

    df = (
        df.withColumn("event_ts", to_timestamp("departure", "yyyy-MM-dd'T'HH:mm:ssXXX"))
          .withColumn("arrival_ts", to_timestamp("arrival", "yyyy-MM-dd'T'HH:mm:ssXXX"))
          .withColumn("ingestion_ts", current_timestamp())
    )

    df = df.filter(col("id").isNotNull() & col("event_ts").isNotNull())

    df = df.withColumn(
        "delay_minutes",
        (col("arrival_ts").cast("long") - col("event_ts").cast("long")) / 60.0
    )

    df = df.withWatermark("event_ts", "1 hour")

    df_agg = (
        df.groupBy(
            window("event_ts", "1 hour")
        ).agg(
            count("*").alias("trips_count"),
            avg("delay_minutes").alias("avg_delay_minutes"),
            lit(0).alias("canceled_trips_count")
        )
    )

    df_agg = (
        df_agg
        .withColumn("dt_partition", to_date(col("window.start")))
        .withColumn("hour", hour(col("window.start")))
        .drop("window")
        .withColumn("area_id", lit("Berlin"))
    )

    return df_agg.select(
        "area_id", "trips_count", "avg_delay_minutes",
        "canceled_trips_count", "dt_partition", "hour"
    )

# 4. FUNKCJE POMOCNICZE 

def write_to_silver_batch(df, table_name, partition_cols):
    spark.sql("CREATE DATABASE IF NOT EXISTS silver")
    (
        df.write
          .mode("append")
          .partitionBy(*partition_cols)
          .saveAsTable(f"silver.{table_name}")
    )

def write_to_silver_stream(batch_df, batch_id, table_name, partition_cols):
    if batch_df.count() > 0:
        write_to_silver_batch(batch_df, table_name, partition_cols)

def read_kafka_or_hdfs(topic, schema, hdfs_path, normalize_func, csv_schema=None):
    try:
        print(f"Próba odczytu z Kafki: {topic}...")
        df = (
            spark.readStream
                 .format("kafka")
                 .option("kafka.bootstrap.servers", "localhost:9092")
                 .option("subscribe", topic)
                 .option("startingOffsets", "earliest")
                 .option("failOnDataLoss", "false") 
                 .load()
                 .select(from_json(col("value").cast("string"), schema).alias("data"))
                 .select("data.*")
        )
        df = normalize_func(df)
        return df, True
    except Exception as e:
        print(f"KAFKA ERROR ({topic}): {e}", file=sys.stderr)
        print(f"Fallback do HDFS: {hdfs_path}...", file=sys.stderr)
        
        effective_schema = csv_schema or schema
        df = (
            spark.read
                 .csv(
                     hdfs_path,
                     header=True,
                     schema=effective_schema,
                     mode="DROPMALFORMED"
                 )
        )
        df = normalize_func(df)
        return df, False

# 5. GŁÓWNA LOGIKA 

# --- NEXTBIKE ---
nextbike_df, streaming_nextbike = read_kafka_or_hdfs(
    "raw.bike",
    next_bike_schema,       
    "nifi_out/bike_*.csv",
    normalize_nextbike,
    csv_schema=next_bike_schema
)

if streaming_nextbike:
    query_nextbike = (
        nextbike_df.writeStream
        .foreachBatch(lambda batch_df, batch_id: write_to_silver_stream(batch_df, batch_id, "nextbike_states", ["dt"]))
        .outputMode("append")
        .option("checkpointLocation", "/chk/silver/nextbike_states")
        .start()
    )
else:
    write_to_silver_batch(nextbike_df, "nextbike_states", ["dt"])

# --- WEATHER ---
weather_df, streaming_weather = read_kafka_or_hdfs(
    "raw.weather",
    weather_schema,
    "nifi_out/weather_*.csv",
    normalize_weather
)

if streaming_weather:
    query_weather = (
        weather_df.writeStream
        .foreachBatch(lambda batch_df, batch_id: write_to_silver_stream(batch_df, batch_id, "weather_observations", ["dt"]))
        .outputMode("append")
        .option("checkpointLocation", "/chk/silver/weather_observations")
        .start()
    )
else:
    write_to_silver_batch(weather_df, "weather_observations", ["dt"])

# --- BVG ---
bvg_df, streaming_bvg = read_kafka_or_hdfs(
    "raw.bvg",
    bvg_schema,
    "nifi_out/trips_*.csv",
    normalize_bvg
)

if streaming_bvg:
    query_bvg = (
        bvg_df.writeStream
        .foreachBatch(lambda batch_df, batch_id: write_to_silver_stream(batch_df, batch_id, "bvg_events", ["dt_partition"]))
        .outputMode("append") 
        .option("checkpointLocation", "/chk/silver/bvg_events")
        .start()
    )
else:
    write_to_silver_batch(bvg_df, "bvg_events", ["dt_partition"])

if streaming_nextbike or streaming_weather or streaming_bvg:

    spark.streams.awaitAnyTermination()
