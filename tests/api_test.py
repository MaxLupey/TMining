import multiprocessing
import time
import unittest

import requests

from apis.api import run_flask


class TestAPI(unittest.TestCase):
    process = None

    @classmethod
    def setUpClass(cls):
        cls.process = multiprocessing.Process(target=run_flask,
                                              args=('./tests/test_data/model.mdl', '127.0.0.1', 5000))
        cls.process.start()
        time.sleep(5)

    def test_predict(self):
        response = requests.get('http://127.0.0.1:5000/model/predict', params={"text": "test text"})
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json())

    def test_visualize(self):
        response = requests.get('http://127.0.0.1:5000/model/visualize', params={"text": "test text"})
        self.assertEqual(response.status_code, 200)

    def test_api_wrong_model(self):
        with self.assertRaises(FileNotFoundError):
            run_flask('./models/wrong_model.mdl', '127.0.0.1', 5050)

    def test_api_wrong_host(self):
        with self.assertRaises(ValueError):
            run_flask('./tests/test_data/model.mdl', 'wrong_host', 500000)

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()


if __name__ == '__main__':
    unittest.main()
