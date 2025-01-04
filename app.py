from flask import Flask, request, render_template
import logging
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Initialize the Flask application
application = Flask(__name__)
app = application

# Set up logging
logging.basicConfig(level=logging.INFO)

# Flag to ensure pipeline setup runs only once
pipeline_initialized = False


@app.before_request
def setup_pipeline():
    """
    Set up the pipeline by triggering data ingestion, transformation, and model training
    before the first request. Ensures it runs only once.
    """
    global pipeline_initialized
    if not pipeline_initialized:
        try:
            logging.info("Starting pipeline setup...")

            # Trigger Data Ingestion
            logging.info("Initiating data ingestion...")
            data_ingestion = DataIngestion()
            train_data, test_data = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed.")

            # Trigger Data Transformation
            logging.info("Initiating data transformation...")
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
            logging.info("Data transformation completed.")

            # Trigger Model Training
            logging.info("Initiating model training...")
            model_trainer = ModelTrainer()
            score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model training completed with R2 score: {score}")

            # Mark the pipeline as initialized
            pipeline_initialized = True
        except Exception as e:
            logging.error(f"Pipeline setup failed: {str(e)}")


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collect data from the form
            data = CustomData(
                Over_Number=request.form.get('overNumber'),
                Runs_Scored_Till_That_Over=request.form.get('runsTillOver'),
                Wickets_Taken_Till_That_Over=request.form.get('wicketsTillOver'),
                Runs_in_Last_5_Overs=request.form.get('runsLast5overs'),
                Wickets_in_Last_5_Overs=request.form.get('wicketsLast5overs'),
                Batting_team=request.form.get('battingteam'),
                Bowling_team=request.form.get('bowlingteam'),
            )

            # Validate data
            if not all([data.Over_Number, data.Runs_Scored_Till_That_Over, data.Wickets_Taken_Till_That_Over,
                        data.Runs_in_Last_5_Overs, data.Wickets_in_Last_5_Overs, data.Batting_team, data.Bowling_team]):
                return render_template('home.html', error="All fields are required!")

            # Process and predict
            pred_df = data.get_data_as_df()
            logging.info(f"Prediction request data: {pred_df}")
            predict_pipeline = PredictionPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('results.html', results=int(results[0]))

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return render_template('home.html', error=f"Error during prediction: {str(e)}")


@app.route('/home')
def home():
    return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
