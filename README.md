# AWS-Spark-Wine-Quality-Prediction-Application

AWS Spark Wine 
Quality Prediction Application
Instruction to use
1. How to run trained win prediction application locally without docker
1. Install Java and Apache spark environment and setup them locally
(https://www.java.com/releases/ )
(https://spark.apache.org/docs/latest)
2. Clone the repository.
3. Navigate inside the project folder in terminal.
4. Place the testing csv file inside “dataset/test/” folder.
5. (Skip this if “classes” folder already has modelPrediciton.class file)
Otherwise, execute following command to generate .class file
“javac --release 16 -cp "{your_apache_spark_jar_folder}/*" -d "classes"
modelPrediction.java”
6. (Skip this if “jar_and_models” folder already has model_pred.jar file)
Otherwise execute following command to generate .jar file
“ jar cf jar_and_models/model_pred.jar -C "classes" . “
7. (NOTE: keep in mind STEP 4)
Execute following command:
“spark-submit --class modelPrediction jar_and_models/model_pred.jar 
{test_file_name}”
2. How to run trained win prediction application using docker
1. Install the docker on your system.
( https://docs.docker.com/tags/release-notes/ )
2. Use docker to pull the application image hosted publicly on docker hub
“docker pull nikhil30113/wine_prediction_app “
3. Place your testfile in any folder (let’s call it DirA)
4. Run the following command
“ docker run -v {DirA complete path}:/app/dataset/test nikhil30113/wine_prediction_app 
{test file name} “
