
import unittest
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from app import app


class StudentResultTest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_valid_prediction(self):
        data = {
            "internal_marks": 85,
            "assignment_marks": 90,
            "previous_score": 88,
            "attendance": 95
        }

        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)

    def test_missing_field(self):
        data = {
            "internal_marks": 85,
            "assignment_marks": 90,
            "previous_score": 88
        }

        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_unknown_route(self):
        response = self.client.get("/unknown")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
