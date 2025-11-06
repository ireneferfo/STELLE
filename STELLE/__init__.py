# """
# Provides concept-based models with STL formula integration for interpretable time series classification.
# """

# from .model.base_model import BaseConceptModel, ForwardWrapper
# from .model.concept_model import ConceptBasedModel
# from .model.model_utils import (
#     compute_attention_jit,
#     concept_based_loss,
#     compute_attention_collapse_regularization,
#     compute_epsilon_control_regularization,
#     ModelCheckpoint,
#     TrainingMonitor,
# )
# from .model.explanation_extractor import ExplanationExtractor

# __all__ = [
#     # Core models
#     'BaseConceptModel',
#     'ConceptBasedModel',
#     'ForwardWrapper',
    
#     # Loss functions and utilities
#     'compute_attention_jit',
#     'concept_based_loss', 
#     'compute_attention_collapse_regularization',
#     'compute_epsilon_control_regularization',
    
#     # Training utilities
#     'ModelCheckpoint',
#     'TrainingMonitor',
    
#     # Explanation generation
#     'ExplanationExtractor',
# ]

# # Version
# __version__ = "1.0.0"