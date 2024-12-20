from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import ModelTraining
from cnnClassifier import logger


STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        trainer_config = config.get_trainer_config()
        trainer = ModelTraining(config=trainer_config)
        trainer.get_updated_base_model()
        trainer.train_valid_generator()
        trainer.train()
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<< \n\n x===========x")
        
    except Exception as e:
        logger.exception(e)
        raise e
        
        