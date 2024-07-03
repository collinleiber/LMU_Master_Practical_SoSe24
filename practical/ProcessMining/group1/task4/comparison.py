from typing import List, Dict
from matplotlib import pyplot as plt

from practical.ProcessMining.group1.task4.tokenreplay import TokenReplay


class ModelComparator:
    """
    Compares multiple TokenReplay instances to find pareto optimal models.

    Attributes:
        model_list: List containing TokenReplay instances, which are based on petri nets
    """
    def __init__(self, model_list: List[TokenReplay]):
        self.model_list = model_list

    def run(self, x_dimension: str = "fitness", y_dimension: str = "precision") -> List[Dict[str: str | float]]:
        """
        Executes the comparison for given models, calculates pareto efficient models and plots the result

        Parameters:
            x_dimension: name or shortcut of the x-axis dimension
            y_dimension: name or shortcut of the y-axis dimension

        Returns:
            List of all models, that are pareto efficient with chosen dimension's scores
        """
        pareto_optima = self._get_pareto_efficient_models(x_dimension, y_dimension)
        result = []

        for model in pareto_optima:
            x_val = model.get_dimension_value(x_dimension)
            y_val = model.get_dimension_value(y_dimension)

            result.append({"type": model.get_discovery_type(), x_dimension: x_val, y_dimension: y_val})

        self.visualize_models(x_dimension, y_dimension)
        return result

    def get_models_values(self) -> List[Dict[str: str | float]]:
        """
        Get values for all dimensions of all models.

        Returns:
            List of all models, with all dimension scores
        """
        result = []

        for model in self.model_list:
            result.append({
                "type": model.get_discovery_type(),
                "f": model.get_fitness(),
                "s": model.get_simplicity(),
                "p": model.get_precision(),
                "g": model.get_generalization()
            })

        return result

    def _get_pareto_efficient_models(self, x_dimension: str, y_dimension: str) -> List[TokenReplay]:
        """
        Get all models which have for the given dimensions no other model that has a higher value for x and y.

        Parameters:
            x_dimension: name or shortcut of the x-axis dimension
            y_dimension: name or shortcut of the y-axis dimension

        Returns:
            List of all models, that are pareto efficient
        """
        pareto_efficient_models = self.model_list.copy()

        for model in self.model_list:
            for other_model in self.model_list:
                if (other_model is not model and
                        (other_model.get_dimension_value(x_dimension) > model.get_dimension_value(x_dimension) and
                         other_model.get_dimension_value(y_dimension) >= model.get_dimension_value(y_dimension)) or
                        (other_model.get_dimension_value(x_dimension) >= model.get_dimension_value(x_dimension) and
                         other_model.get_dimension_value(y_dimension) > model.get_dimension_value(y_dimension))):
                    if model in pareto_efficient_models:
                        pareto_efficient_models.remove(model)
                    break

        return pareto_efficient_models

    def visualize_models(self, x_dimension: str, y_dimension: str):
        """
        plots model comparison and highlights from all models the pareto efficient one's in red

        Parameters:
            x_dimension: name or shortcut of the x-axis dimension
            y_dimension: name or shortcut of the y-axis dimension
        """
        pareto_efficient_models = self._get_pareto_efficient_models(x_dimension, y_dimension)

        for model in self.model_list:
            x_val = model.get_dimension_value(x_dimension)
            y_val = model.get_dimension_value(y_dimension)
            color = 'red' if model in pareto_efficient_models else 'blue'
            plt.scatter(x_val, y_val, color=color)
            plt.text(x_val, y_val, model.get_discovery_type(), fontsize=9)

        plt.xlabel(x_dimension)
        plt.ylabel(y_dimension)
        plt.show()
