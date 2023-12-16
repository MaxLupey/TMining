import argparse
import os.path

from apis.api import run_flask
from apis.model import App


def make_dir(dir_path: str) -> None:
    """
    Creates a directory if it does not exist.

    Parameters
    ----------
    dir_path : str
        Path to the directory.
    """
    if dir_path is not None and not os.path.exists(os.path.dirname(dir_path)):
        os.makedirs(os.path.dirname(dir_path))


def main():
    """
    Main function to run the application.
    It parses the command line arguments and calls the appropriate function based on the command.
    """
    path_constant = "Path to the dataset"
    path_to_model = "Path to the trained model"
    x_constant = "Name of the column with the text"
    y_constant = "Name of the column with the target"
    model_constant = "./model.mdl"

    # Create a parser for command line arguments
    parser = argparse.ArgumentParser(description="Console App")
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    # Parser for 'train' command
    parser_train = subparsers.add_parser('train', help='Train a new model and save it to a file',
                                         description='Train a new machine learning model and save it to a file.')
    parser_train.add_argument('-dataset_path', help=path_constant, metavar="./data/factcheck.csv", required=True,
                              type=str)
    parser_train.add_argument('-x', help=x_constant, default="text", metavar="text", type=str)
    parser_train.add_argument('-y', help=y_constant, default="target", metavar=" target", type=str)
    parser_train.add_argument('-kfold', help="Number of folds for cross validation", default=1, metavar="10",
                              type=int)
    parser_train.add_argument('-test_size', help="Size of the test set", default=0, metavar="0.2", type=float)
    parser_train.add_argument('-save_to', help="Path to save the model", metavar="./result", type=str)
    parser_train.add_argument('-model', help="Name of the model", default="SVC", metavar="SVC", type=str)
    parser_train.add_argument('-vectorizer', help="Name of the vectorizer", default="TfidfVectorizer",
                              metavar='TfidfVectorizer', type=str)

    # Parser for 'validate' command
    parser_validate = subparsers.add_parser('validate', help='Validates the accuracy and f1 of a trained model.',
                                            description='Validates the accuracy and f1 of a trained model.')
    parser_validate.add_argument('-model_path', help=path_to_model, metavar=model_constant, type=str, required=True)
    parser_validate.add_argument('-dataset_path', help=path_constant, metavar="./data/factcheck.csv", type=str,
                                 required=True)
    parser_validate.add_argument('-x', help=x_constant, default="text", metavar="text", type=str)
    parser_validate.add_argument('-y', help=y_constant, default="target", metavar="target", type=str)
    parser_validate.add_argument('-test_size', help="Size of the test set", default=0.2, metavar="0.2", type=float)

    # Parser for 'predict' command
    parser_predict = subparsers.add_parser('predict', help='Make a prediction for a given text',
                                           description='Make predictions for input text using a trained model.')
    parser_predict.add_argument('-model_path', help=path_to_model, metavar=model_constant, type=str, required=True)
    parser_predict.add_argument('-text', help="Text to predict", metavar="fake news text", type=str, required=True)

    # Parser for 'visualize' command
    parser_visualize = subparsers.add_parser('visualize', help='Visualize the prediction',
                                             description='Visualize the predictions made by the model.')
    parser_visualize.add_argument('-model_path', help=path_to_model, metavar=model_constant, required=True)
    parser_visualize.add_argument('-text', help="Text to predict for visualization", metavar="fake news text",
                                  type=str)
    parser_visualize.add_argument('-features', help="Number of features to visualize", metavar="60", type=int,
                                  default=40)
    parser_visualize.add_argument('-save_to', help="Path to save prediction results", metavar="./result",
                                  default="./results/1.html", type=str)

    # Parser for 'host' command
    parser_host = subparsers.add_parser('host', help='Host model as a REST API',
                                        description='Run the model as a REST-ful API service.')
    parser_host.add_argument('-model_path', help=path_to_model, metavar=model_constant, type=str, required=True)
    parser_host.add_argument('-address', help="Host to run the API on", metavar="0.0.0.0", default="0.0.0.0", type=str)
    parser_host.add_argument('-port', help="Port to run the API on", metavar="5000", default=5000, type=int)

    # Parse the command line arguments
    args = parser.parse_args()
    print(args)
    # Create an instance of the App class
    app = App()
    # Call the appropriate function based on the command
    try:
        if args.command == 'train':
            make_dir(args.save_to)
            model = app.train_model(dataset_path=args.dataset_path, x=args.x, y=args.y, kfold=int(args.kfold),
                                    test_size=float(args.test_size), save=True if args.save_to else False,
                                    model=args.model, vectorizer=args.vectorizer)
            import joblib
            if model is not None:
                joblib.dump(model, args.save_to)
                print(f"Model saved in file: {args.save_to}")
        elif args.command == 'predict':
            print(f"Prediction result: {app.predict(model_path=args.model_path, text=args.text)}")
        elif args.command == 'visualize':
            import io
            visualization = (app.visualize(model_path=args.model_path, text=args.text, num_features=int(args.features)))
            make_dir(args.save_to)
            with io.open(args.save_to, 'w', encoding='utf-8') as f:
                f.write(str(visualization))
            print(f"Visualization saved to file: {args.save_to}")
        elif args.command == 'host':
            run_flask(model_path=args.model_path, address=args.address, port=args.port)
        elif args.command == 'validate':
            accuracy, f1 = app.validate(dataset_path=args.dataset_path, model_path=args.model_path, x=args.x, y=args.y,
                                        size=float(args.test_size))
            print(f"Accuracy: {accuracy}, F1: {f1}")
        else:
            print("No valid command provided. Use --help for usage information.")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(e)


if __name__ == "__main__":
    main()
