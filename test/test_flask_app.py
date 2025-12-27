import unittest
from app import create_app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.testing = True
        cls.client = cls.app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_predict_page(self):
        response = self.client.post('/predict', data={"text": "I love this!"})
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
