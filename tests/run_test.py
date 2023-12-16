import argparse
import unittest
from unittest.mock import patch

import main


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = './tests/test_data/dataset.csv'
        self.model = './tests/test_data/model.mdl'
        self.test_text = 'This is a test text'
        self.save_result = './tests/test_data/result.html'
        self.invalid_model = 'invalid_model.mdl'

    @patch('argparse.ArgumentParser.parse_args')
    def test_main_train(self, mock_args):
        mock_args.return_value = argparse.Namespace(command='train', dataset_path=self.dataset, svm='SVC',
                                                    vectorizer='TfidfVectorizer', x='text', y='target', kfold=1,
                                                    size=0, save=self.model)
        main.main()

    @patch('argparse.ArgumentParser.parse_args')
    def test_main_predict(self, mock_args):
        mock_args.return_value = argparse.Namespace(command='predict', model_path=self.model, text=self.test_text)
        main.main()

    @patch('argparse.ArgumentParser.parse_args')
    def test_main_visualize(self, mock_args):
        mock_args.return_value = argparse.Namespace(command='visualize', model_path=self.model, text=self.test_text,
                                                    features=40, save=self.save_result)
        main.main()

    @patch('argparse.ArgumentParser.parse_args')
    def test_main_validate(self, mock_args):
        mock_args.return_value = argparse.Namespace(command='validate', model_path=self.model,
                                                    dataset_path=self.dataset, x='text', y='target', size=0.2)
        main.main()


if __name__ == '__main__':
    unittest.main()
