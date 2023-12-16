import unittest

from apis.model import App


class TestModel(unittest.TestCase):
    def setUp(self):
        self.app = App()
        self.invalid_dataset_constant = 'invalid_dataset.csv'
        self.invalid_model_constant = 'invalid_model.mdl'
        self.dataset_path_constant = './tests/test_data/dataset.csv'
        self.model_path_constant = './tests/test_data/model.mdl'

    def test_train_model(self):
        model = (self.app.train_model(dataset_path=self.dataset_path_constant, save=True))
        self.assertIsNotNone(model)

    def test_train_model_invalid_data(self):
        with self.assertRaises(FileNotFoundError):
            self.app.train_model(dataset_path=self.invalid_dataset_constant, save=True)

    def test_predict(self):
        prediction = str(self.app.predict(self.model_path_constant, 'This text was tested by svc'))
        self.assertIsNotNone(prediction)

    def test_predict_invalid_model(self):
        with self.assertRaises(FileNotFoundError):
            self.app.predict(self.invalid_model_constant, 'This text for invalid model')

    def test_predict_empty_text(self):
        with self.assertRaises(ValueError):
            self.app.predict(self.model_path_constant, '')

    def test_visualize(self):
        visualization = (self.app.visualize(self.model_path_constant, 'This text was written for visualization '
                                                                      'prediction'))
        self.assertIsNotNone(visualization)

    def test_visualize_invalid_model(self):
        with self.assertRaises(FileNotFoundError):
            self.app.visualize(self.invalid_model_constant, 'This text for invalid model')

    def test_visualize_empty_text(self):
        with self.assertRaises(ValueError):
            self.app.visualize(self.model_path_constant, '')

    def test_validate(self):
        result = self.app.validate(model_path=self.model_path_constant, dataset_path=self.dataset_path_constant,
                                   x='text', y='target', size=0.2)
        self.assertIsNotNone(result)

    def test_validate_invalid_parameters(self):
        with self.assertRaises(FileNotFoundError):
            self.app.validate(self.invalid_model_constant, self.dataset_path_constant, 'text', 'target', 0.2)
        with self.assertRaises(FileNotFoundError):
            self.app.validate(self.model_path_constant, self.invalid_dataset_constant, 'text', 'target', 0.2)
        with self.assertRaises(ValueError):
            self.app.validate(self.model_path_constant, self.dataset_path_constant, 'invalid_column', 'invalid_column',
                              0.2)
        with self.assertRaises(ValueError):
            self.app.validate(self.model_path_constant, self.dataset_path_constant, 'text', 'target', 2.0)
        with self.assertRaises(ValueError):
            self.app.validate(self.model_path_constant, self.dataset_path_constant, 'text', 'target', -1.0)


if __name__ == '__main__':
    unittest.main()
