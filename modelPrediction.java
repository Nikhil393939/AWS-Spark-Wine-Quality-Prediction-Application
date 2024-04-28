
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.types.DataTypes;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;


public class modelPrediction {

    public static Dataset<Row> dataCleaning(Dataset<Row> df) {
        String[] columns = df.columns();
        for (String column : columns) {
            String cleanedColumnName = column.replace("\"", "").trim();
            df = df.withColumnRenamed(column, cleanedColumnName); // Rename column
            df = df.withColumn(cleanedColumnName, col(cleanedColumnName).cast("double"));
        }
        return df;
    }

    public static Dataset<Row> loadData(SparkSession spark, String filepath)
    {
        Dataset<Row> raw_data = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .csv(filepath);

        return dataCleaning(raw_data);
    }

    public static void main(String[] args) {
        if (args.length != 1)
        {
            System.err.println("Usage: spark-submit --class modelPrediction jar_and_models/model_pred.jar <filename>");
            return;
        }
        
        String testFile = "dataset/test/" + args[0];
        String saveModelPath = "jar_and_models/best_model";
        
        // Initialize Spark
        JavaSparkContext sc = new JavaSparkContext();
        SparkSession spark = SparkSession.builder().getOrCreate();
        
        // Load data (assuming you have validation data)
        Dataset<Row> validation_data = loadData(spark, testFile);
        
        // Load the saved model
        CrossValidatorModel loadedModel = CrossValidatorModel.load(saveModelPath);
        
        // Make predictions using the loaded model
        Dataset<Row> predictions = loadedModel.transform(validation_data);
        
        // Perform evaluation if needed
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("quality")
        .setPredictionCol("prediction")
        .setMetricName("f1");
        
        double f1_score = evaluator.evaluate(predictions);
        
        System.out.println("F1 score on validation data = " + f1_score);

        // Stop Spark
        spark.stop();
    }
}
