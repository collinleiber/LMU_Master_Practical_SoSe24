import unittest
from practical.ProcessMining.group1.task4.comparison import ModelComparator


class TestModelComparator(unittest.TestCase):

    def setUp(self):
        self.model1 = self.create_mock_model("Model 1", 0.9, 0.7, 0.8, 0.6)
        self.model2 = self.create_mock_model("Model 2", 0.85, 0.75, 0.88, 0.65)
        self.model3 = self.create_mock_model("Model 3", 0.7, 0.85, 0.6, 0.7)

        self.model_list = [self.model1, self.model2, self.model3]
        self.comparator = ModelComparator(self.model_list)

    def create_mock_model(self, discovery_type, fitness, simplicity, precision, generalization):
        model = type('MockTokenReplay', (), {})()
        model.get_discovery_type = lambda: discovery_type
        model.get_fitness = lambda: fitness
        model.get_simplicity = lambda: simplicity
        model.get_precision = lambda: precision
        model.get_generalization = lambda: generalization
        model.get_dimension_value = lambda dimension: {
            "fitness": fitness,
            "simplicity": simplicity,
            "precision": precision,
            "generalization": generalization
        }[dimension]
        return model

    def test_get_models_values(self):
        expected_values = [
            {"type": "Model 1", "f": 0.9, "s": 0.7, "p": 0.8, "g": 0.6},
            {"type": "Model 2", "f": 0.85, "s": 0.75, "p": 0.88, "g": 0.65},
            {"type": "Model 3", "f": 0.7, "s": 0.85, "p": 0.6, "g": 0.7}
        ]
        models_values = self.comparator.get_models_values()
        self.assertEqual(models_values, expected_values)

    def test_run(self):
        expected_result = [
            {"type": "Model 1", "fitness": 0.9, "precision": 0.8},
            {"type": "Model 2", "fitness": 0.85, "precision": 0.88}
        ]
        result = self.comparator.run(x_dimension="fitness", y_dimension="precision")
        self.assertEqual(result, expected_result)

    def test_get_pareto_efficient_models(self):
        pareto_models = self.comparator._get_pareto_efficient_models("fitness", "precision")
        self.assertEqual(pareto_models, [self.model1, self.model2])

    def test_visualize_models(self):
        self.comparator.visualize_models("fitness", "precision")