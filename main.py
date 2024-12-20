from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

# stage 1: data ingestion
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<< \n\n")
except Exception as e:
    logger.exception(e)
    raise e


# stage 2: prepare base model
STAGE_NAME = "Prepare base model stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<< \n\n")
except Exception as e:
    logger.exception(e)
    raise e


# stage 3: Model training
STAGE_NAME = "Model Training stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<< \n\n")
    
except Exception as e:
    logger.exception(e)
    raise e


# stage 4: Model evaluation and tracking with mlflow
STAGE_NAME = "Model evaluation stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<< \n\n")

except Exception as e:
    logger.exception(e)
    raise e