from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig
import os


class ConfigurationManager:
    def __init__(self, config_path = CONFIG_FILE_PATH, params_path = PARAMS_FILE_PATH, secret_path = SECRET_FILE_PATH):

        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.secret = read_yaml(secret_path)
        

        create_directories([self.config.artifacts_root])
    
    # get data ingestion config
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    # get base model config and params
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learing_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    
    # get the training config and params
    def get_trainer_config(self) -> TrainingConfig:
        training = self.config.model_training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest-CT-Scan-data")

        create_directories([
            Path(training.root_dir)
        ])

        trainer_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return trainer_config
    
    #Evaluation config
    def get_evaluation_config(self) -> EvaluationConfig:
        model_training_config = self.config.model_training
        data_ingestion_config = self.config.data_ingestion
        mlflow_credentials = self.secret.mlflow_tracking_credentials
        
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_credentials.MLFLOW_TRACKING_URI
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_credentials.MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_credentials.MLFLOW_TRACKING_PASSWORD
                
        evaluation_config = EvaluationConfig(
            path_of_the_model=Path(model_training_config.trained_model_path),
            training_data=Path(data_ingestion_config.training_data_path),
            all_params=self.params,
            mlflow_uri=mlflow_credentials.MLFLOW_TRACKING_URI,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        
        return evaluation_config


