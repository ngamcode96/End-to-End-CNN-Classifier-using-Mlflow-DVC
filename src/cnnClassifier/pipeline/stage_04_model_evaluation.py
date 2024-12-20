from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation import ModelEvaluation
from cnnClassifier import logger

STAGE_NAME = "Model evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        evauation_config = config.get_evaluation_config()
        model_evaluation = ModelEvaluation(evauation_config)
        model_evaluation.evaluation()
        model_evaluation.save_score()
        model_evaluation.log_into_mlflow()
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<< \n\n")
        
    except Exception as e:
        logger.exception(e)
        raise e

    
    
    