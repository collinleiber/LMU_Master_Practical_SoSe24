from typing import List, Dict

from practical.ProcessMining.group1.task4.tokenreplay import TokenReplay


class ModelComparator:
    """
    Compares multiple TokenReplay instances to find pareto optimal models.
    """
    def __init__(self, model_list: List[TokenReplay]):
        self.model_list = model_list

    def run(self, x_dimension: str = "fitness", y_dimension: str = "precision") -> Dict:
        pareto_optima = self._get_pareto_efficient_models(x_dimension, y_dimension)
        result = {}

        for index, model in enumerate(pareto_optima):
            x_val = model.get_dimension_value(x_dimension)
            y_val = model.get_dimension_value(y_dimension)

            result[index] = {"type": model.get_discovery_type(), x_dimension: x_val, y_dimension: y_val}

        return result

    def get_models_values(self) -> Dict:
        """ Get all dimensions of all models. """
        result = {}

        for index, model in enumerate(self.model_list):
            result[index] = {"type": model.get_discovery_type(),
                             "f": model.get_fitness(),
                             "s": model.get_simplicity(),
                             "p": model.get_precision(),
                             "g": model.get_generalization()}

        return result

    def _get_pareto_efficient_models(self, x_dimension: str, y_dimension: str) -> List[TokenReplay]:
        """
        Get all models which have for the given dimensions no other model that has a higher value for x and y.

        Parameters:
            x_dimension: name or shortcut of the x-axis dimension
            y_dimension: name or shortcut of the y-axis dimension

        Returns:
            List of all models, that are pareto
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