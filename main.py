import argparse

from apis.api import run_flask
from apis.model import App


def main() -> object:
    """
    Main function to run the application.
    It parses the command line arguments and calls the appropriate function based on the command.
    """
    # Create a parser for command line arguments
    parser = argparse.ArgumentParser(description="Console App")
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    # Parser for 'train' command
    parser_train = subparsers.add_parser('train', help='Train a new model and save it to a file',
                                         description='Train a new machine learning model and save it to a file.')
    parser_train.add_argument('-path', help="Path to the dataset", metavar="path")
    parser_train.add_argument('-svm', help="Name of the model", default="SVC", metavar="model")
    parser_train.add_argument('-vectorizer', help="Name of the vectorizer", default="TfidfVectorizer",
                              metavar="vectorizer")
    parser_train.add_argument('-x', help="Name of the column with the text", default="text", metavar="data_x")
    parser_train.add_argument('-y', help="Name of the column with the target", default="target", metavar="data_y")
    parser_train.add_argument('-save', help="Path to save the model", default="./model.mdl", metavar="save_to")

    # Parser for 'predict' command
    parser_predict = subparsers.add_parser('predict', help='Make a prediction for a given text',
                                           description='Make predictions for input text using a trained model.')
    parser_predict.add_argument('-path', help="Model path", metavar="model_path")
    parser_predict.add_argument('-text', help="Text to predict", metavar="text")

    # Parser for 'visualize' command
    parser_visualize = subparsers.add_parser('visualize', help='Visualize the prediction',
                                             description='Visualize the predictions made by the model.')
    parser_visualize.add_argument('-path', help="Model path", metavar="model_path")
    parser_visualize.add_argument('-text', help="Text to predict for visualization", metavar="text")
    parser_visualize.add_argument('-features', help="Number of features to visualize", metavar="num_features",
                                  default=40)
    parser_visualize.add_argument('-save', help="Path to save prediction results", metavar="save_to",
                                  default="./results/1.html")

    # Parser for 'host' command
    parser_host = subparsers.add_parser('host', help='Host model as a REST API',
                                        description='Run the model as a REST-ful API service.')
    parser_host.add_argument('-model', help="Path to the model", metavar="model_path")
    parser_host.add_argument('-address', help="Host to run the API on", metavar="host", default="0.0.0.0")
    parser_host.add_argument('-port', help="Port to run the API on", metavar="port", default="5000")

    # Parse the command line arguments
    args = parser.parse_args()

    # Create an instance of the App class
    app = App()

    # Call the appropriate function based on the command
    if args.command == 'train':
        if args.path is None:
            print("No dataset provided. Use --help for usage information.")
            return
        app.train_model(dataset_path=args.path, text=args.x, target=args.y, save_to=args.save, model=args.svm,
                        vectorizer=args.vectorizer)
    elif args.command == 'predict':
        if args.path is None or args.text is None:
            print("No command options provided. Use --help for usage information.")
            return
        app.predict(model_path=args.path, text=args.text)

    elif args.command == 'visualize':
        if args.path is None or args.text is None:
            print("No command options provided. Use --help for usage information.")
            return
        app.visualize(model_path=args.path, text=args.text, num_features=int(args.features), save_to=args.save)
    elif args.command == 'host':
        if args.model is None:
            print("No model provided. Use --help for usage information.")
            return
        run_flask(model_path=args.model, address=args.address, port=args.port)
    else:
        print("No valid command provided. Use --help for usage information.")


if __name__ == "__main__":
    main()
