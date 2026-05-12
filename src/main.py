from services.classifier import Classifier
from services.evaluator import Evaluator
from services.eda_service import EDAService
from utils.kaggle_hub import KaggleHub
from services.dataset_indexer import DatasetIndexer
from services.imagePreprocessor import ImagePreprocessor
from services.data_processor import DataProcessor
from utils.visualizer import Visualizer
from config import RAW_DATA_DIR, MODEL_OUTPUT_DIR, EDA_OUTPUT_DIR

def main():
    while True:
        # main menu
        print("1 - download data")
        print("2 - index dataset")
        print("3 - begin EDA")
        print("4 - pre-train model")
        print("5 - predict single image")
        print("6 - print EDA summary")
        print("7 - eda visualizations")
        print("q - quit")

        picker = input("enter selection:")

        if picker == "1":
            # Step 1: Download data
            print("Getting data...")
            KaggleHub(RAW_DATA_DIR).get_raw_data()
       
        elif picker == "2":
            # Step 2: Index dataset
            print("Indexing dataset...")
            indexer = DatasetIndexer()
            df = indexer.validate_dataframe(indexer.build_dataframe())
            indexer.save_indexed_data(df)
            indexer.save_sample_grid(df)
        
        elif picker == "3":
            if df is None:
                print("Please index the dataset first (option 2).")
                continue
            print("Generating EDA outputs...")
            eda = EDAService(df, EDA_OUTPUT_DIR)
            eda.save_class_distribution()
            eda.save_image_size_distribution()
            summary = eda.build_summary()
            print("EDA summary built.")

        elif picker == "4":
            if df is None:
                print("Please index the dataset first (option 2).")
                continue
            # Step 4: Preprocess images
            print("Preprocessing images...")
            preprocessor = ImagePreprocessor()
            X, y = preprocessor.transform_batch(df)
            
            # Step 5: Load and split data
            print("Loading and splitting data...")
            data_processor = DataProcessor(df, X, y)
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.split_data()

            # Step 6: Train model
            print("Training model...")
            model_trainer = Classifier(preprocessor, MODEL_OUTPUT_DIR)
            results = model_trainer.train(df)
            # print(results)
            model_trainer.save_model()

            # # Step 7: Evaluate model
            # print("Evaluating model...")
            # y_pred = model_trainer.predict(X_test)
            # eval_obj = Evaluator()
            # metrics = eval_obj.evaluate(y_test, y_pred)
            # eval_obj.save_metrics(metrics)
            # labels = loader.get_labels()
            # eval_obj.plot_confusion_matrix(y_test, y_pred, labels)
        elif picker == "5":

            print("Loading model...")

            preprocessor = ImagePreprocessor()

            classifier = Classifier(preprocessor, MODEL_OUTPUT_DIR)

            classifier.load_model()

            image_path = input("Enter image path: ")

            prediction, confidence = classifier.predict_single(image_path)

            print(f"Prediction: {prediction}")

            print(f"Confidence: {confidence:.2%}")


        elif picker == "6":
            if summary is None:
                print("Please generate EDA outputs first (option 3).")
                continue
            else:
                print("EDA summary:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")

        elif picker == "7":
            # Step 8: Visualize (optional)
            print("Generating visualizations...")
            viz = Visualizer()
            viz.plot_class_distribution(df)
        
        elif picker.lower() == "q":
            print("Exiting...")
            break
        
        else:
            print("pick a valid option")

            

    


    

    

    
    
    print("Pipeline completed!")

if __name__ == "__main__":
    main()
