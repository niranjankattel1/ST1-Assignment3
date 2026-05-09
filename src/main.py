from services.classifier import Classifier
from utils.kaggle_hub import KaggleHub
from services.dataset_indexer import DatasetIndexer
from services.imagePreprocessor import ImagePreprocessor
from services.data_processor import DataProcessor
from config import RAW_DATA_DIR, MODEL_OUTPUT_DIR

def main():

    # Step 1: Download data
    print("Getting data...")
    KaggleHub(RAW_DATA_DIR).get_raw_data()

    # Step 2: Index dataset
    print("Indexing dataset...")
    indexer = DatasetIndexer()
    df = indexer.validate_dataframe(indexer.build_dataframe())
    indexer.save_indexed_data(df)
    indexer.save_sample_grid(df)


    # # Step 4: Preprocess images
    # print("Preprocessing images...")
    preprocessor = ImagePreprocessor()
    # X, y = preprocessor.transform_batch(df)
    #
    # # Step 5: Load and split data
    # print("Loading and splitting data...")
    # data_processor = DataProcessor(df, X, y)
    # (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.split_data()

    # Step 6: Train model
    print("Training model...")
    model_trainer = Classifier(preprocessor, MODEL_OUTPUT_DIR)
    model_trainer.train(df)
    model_trainer.save_model()

    # # Step 7: Evaluate model
    # print("Evaluating model...")
    # y_pred = model_trainer.predict(X_test)
    # eval_obj = Evaluator()
    # metrics = eval_obj.evaluate(y_test, y_pred)
    # eval_obj.save_metrics(metrics)
    # labels = loader.get_labels()
    # eval_obj.plot_confusion_matrix(y_test, y_pred, labels)

    # # Step 8: Visualize (optional)
    # print("Generating visualizations...")
    # viz = Visualizer()
    # viz.plot_class_distribution(df)
    #
    # print("Pipeline completed!")

if __name__ == "__main__":
    main()
