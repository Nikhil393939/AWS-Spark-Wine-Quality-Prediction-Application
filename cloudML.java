
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


public class cloudML {

    private static String trainFilePath = "dataset/train/TrainingDataset.csv";
    private static String validationFilepath = "dataset/train/ValidationDataset.csv";
    private static String saveModelPath = "jar_and_models/best_model";

    // Clean column names and cast columns to double
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

        // Initialize Spark
        JavaSparkContext sc = new JavaSparkContext();
        SparkSession spark = SparkSession.builder().getOrCreate();

        // Load data
        Dataset<Row> train_data = loadData(spark, trainFilePath);
        Dataset<Row> validation_data = loadData(spark, validationFilepath);
        

        String[] featureColumns = new String[]{"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"};
        String labelColumn = "quality";


        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol(labelColumn)
                .setFeaturesCol("features");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features");
        
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{new StringIndexer()
                    .setInputCol(labelColumn)
                    .setOutputCol("indexedLabel"),assembler,
                    rf}
                );

        // Set up parameter grid for tuning
        ParamGridBuilder paramGrid = new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[]{10, 20, 30})
                .addGrid(rf.maxDepth(), new int[]{5, 10, 15});

        // Set up cross-validation
        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new RegressionEvaluator().setLabelCol(labelColumn))
                .setEstimatorParamMaps(paramGrid.build())
                .setNumFolds(5);

        // Train model
        CrossValidatorModel cvModel = cv.fit(train_data);

        // Make predictions
        Dataset<Row> predictions = cvModel.transform(validation_data);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol(labelColumn)
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1_score = evaluator.evaluate(predictions);
        
        // save the model
        try{
            cvModel.write().overwrite().save(saveModelPath);
        }
        catch(Exception e){
            e.printStackTrace();
        }
        
        System.out.println("F1 score on validation data = " + f1_score);
        // Stop Spark
        spark.stop();
    }
}
