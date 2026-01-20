from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, current_timestamp, round,
    avg, sum as sum_, coalesce, lit, row_number
)
from pyspark.sql.window import Window
import os

def create_spark():
    spark = (
        SparkSession.builder
        .appName("GoldSpark_BikeWeatherBvg")
        .enableHiveSupport()
        # Wyłączenie algorytmu zamykania plików, który rzuca błędem
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        # WYŁĄCZENIE SUM KONTROLNYCH (To zastępuje RawLocalFileSystem w bezpieczny sposób)
        .config("spark.hadoop.fs.file.impl.disable.cache", "true")
        # Ignorowanie problemów z uprawnieniami do mkdirs na lokalnym FS
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.shuffle.partitions", "5")
        .getOrCreate()
    )
    # Dodatkowo wymuszamy na Hadoopie brak sprawdzania sum kontrolnych na poziomie sesji
    spark.sparkContext._jsc.hadoopConfiguration().set("fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    
    spark.sparkContext.setLogLevel("WARN")
    return spark
def load_silver_tables(spark):
    nextbike = spark.table("silver.nextbike_states")
    weather = spark.table("silver.weather_observations")
    bvg = spark.table("silver.bvg_events")
    return nextbike, weather, bvg

def build_gold(nextbike, weather, bvg):
    # 1. Deduplikacja Nextbike (bierzemy ostatni stan w danej godzinie)
    nb = (
        nextbike
        .withColumn(
            "rn",
            row_number().over(
                Window.partitionBy("station_id", "dt", "hour")
                    .orderBy(col("events_ts").desc())
            )
        )
        .filter(col("rn") == 1)
        .drop("rn")
        .select(
            "station_id", "station_name", "area_id", "lat", "lon",
            "events_ts", "dt", "hour",
            "bikes_available", "slots_available", "is_reserved", "is_disabled"
        )
    )

    # 2. Agregacja Pogody
    w_agg = (
        weather
        .groupBy("dt", "hour")
        .agg(
            avg(col("temperature_avg").cast("double")).alias("temperature_avg"),
            avg(col("wind_speed_avg").cast("double")).alias("wind_speed_avg"),
            sum_(col("precipitation_sum").cast("double")).alias("precipitation_sum")
        )
    )

    # 3. Agregacja BVG
    b_agg = (
        bvg
        .withColumnRenamed("dt_partition", "dt")
        .groupBy("dt", "hour")
        .agg(
            sum_(col("trips_count").cast("long")).alias("trips_count"),
            avg(col("avg_delay_minutes").cast("double")).alias("avg_delay_minutes"),
            sum_(col("canceled_trips_count").cast("long")).alias("canceled_trips_count")
        )
    )

    # 4. Łączenie
    nb_w = nb.join(w_agg, on=["dt", "hour"], how="left")
    nb_w_b = nb_w.join(b_agg, on=["dt", "hour"], how="left")

    # 5. Feature Engineering
    denom = (coalesce(col("bikes_available"), lit(0)) + coalesce(col("slots_available"), lit(0)))

    gold = (
        nb_w_b
        .withColumn(
            "occupancy_ratio",
            when(denom > 0, col("bikes_available") / denom).otherwise(lit(None).cast("double"))
        )
        .withColumn("occupancy_ratio", round(col("occupancy_ratio"), 3))
        .withColumn("gold_ingestion_ts", current_timestamp())
        .select(
            "station_id", "station_name", "area_id", "lat", "lon",
            "events_ts", "dt", "hour",
            "bikes_available", "slots_available", "is_reserved", "is_disabled",
            "occupancy_ratio",
            "temperature_avg", "wind_speed_avg", "precipitation_sum",
            "trips_count", "avg_delay_minutes", "canceled_trips_count",
            "gold_ingestion_ts",
        )
    )

    return gold

def write_gold(spark, gold_df, table_name="bike_weather_bvg_features"):
    spark.sql("CREATE DATABASE IF NOT EXISTS gold")
    
    # Usuwamy tabelę z metastore, żeby uniknąć konfliktów
    spark.sql(f"DROP TABLE IF EXISTS gold.{table_name}")
    
    print(f"Zapisuję tabelę gold.{table_name} (Format: Parquet, Committer v2)...")
    
    (
        gold_df
        .write
        .mode("overwrite")
        .partitionBy("dt")
        .format("parquet")
        .saveAsTable(f"gold.{table_name}")
    )

if __name__ == "__main__":
    # Ręczne czyszczenie plików przed startem
    print("Czyszczenie starego folderu warehouse...")
    os.system("rm -rf /home/vagrant/spark-warehouse/gold.db/bike_weather_bvg_features")

    spark = create_spark()
    
    try:
        nextbike, weather, bvg = load_silver_tables(spark)
        gold_df = build_gold(nextbike, weather, bvg)
        write_gold(spark, gold_df)
        print("SUKCES: Tabela gold.bike_weather_bvg_features została utworzona.")
        
        # Opcjonalnie: Szybkie sprawdzenie czy są dane
        count = spark.table("gold.bike_weather_bvg_features").count()
        print(f"Liczba wierszy w tabeli Gold: {count}")
        
    except Exception as e:
        print(f"BŁĄD PODCZAS PRZETWARZANIA: {e}")
    finally:
        spark.stop()