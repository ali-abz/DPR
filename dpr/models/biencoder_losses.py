import torch
import torch.nn.functional as F
from torch import Tensor as T
# from IPython import embed

import logging


def check_list(name, input_value, expected_values):
    if input_value not in expected_values:
        error_str = f'{name} is expected to be in {expected_values}, got: {input_value}'
        raise ValueError(error_str)


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


_reduction_methods = {
    'sum': torch.sum,
    'mean': torch.mean,
    'none': lambda x: x,
}

logger = logging.getLogger(__name__)


@singleton
class A3GLoss:
    def __init__(self, reduction='mean', version='v1', min_shift=False, cosine_scaler=True):
        check_list('reduction', reduction, ['mean', 'sum', 'none'])
        check_list('version', version, ['v1', 'v2'])
        self.reduction = reduction
        self.version = version
        self.min_shift = min_shift
        self.cosine_scaler = cosine_scaler
        self.reduction_method = _reduction_methods.get(self.reduction)
        ndcg_to_ndcg_loss_methods = {
            'v1': self.ndcg_to_ndcg_loss_v1,
            'v2': self.ndcg_to_ndcg_loss_v2,
        }
        self.ndcg_to_ndcg_loss_functoin = ndcg_to_ndcg_loss_methods.get(self.version)
        self.log_instance()

    def log_instance(self):
        logger.info('Loss function: A3GLoss')
        logger.info(f'A3GLoss reduction: {self.reduction}')
        logger.info(f'A3GLoss version: {self.version}')
        logger.info(f'A3GLoss cosine_scaler: {self.cosine_scaler}')

    def _NDCG(self, ranks: T):
        positions = torch.stack([torch.log2(x) for x in torch.arange(2., len(ranks) + 2)])
        positions = positions.to(ranks.device)
        if self.min_shift:
            min_value = abs(min(ranks.min().detach(), 0))
            ranks += min_value
        dcg = (ranks / positions).sum()
        sorted_ranks = torch.stack(sorted(ranks, reverse=True))
        ideal_dcg = (sorted_ranks / positions).sum()
        ndcg = dcg / ideal_dcg
        return ndcg

    @staticmethod
    def ndcg_to_ndcg_loss_v1(ndcg):
        ndcg_loss = 1 - ndcg
        return ndcg_loss

    @staticmethod
    def ndcg_to_ndcg_loss_v2(ndcg):
        eps = 0.00001
        ndcg_loss = (1 / (ndcg + eps)) - (1 / (1 + eps))
        return ndcg_loss

    def calc(self, scores: T, relations: T):
        if self.cosine_scaler:
            scores = 0.5 * (1 + scores)
        losses = []
        for q_scores, q_relations in zip(scores, relations):
            sorted_q_scores, sorted_q_relations = zip(*sorted(zip(q_scores, q_relations), reverse=True))
            sorted_q_scores = torch.stack(sorted_q_scores)
            sorted_q_relations = torch.stack(sorted_q_relations)
            ndcg = self._NDCG(sorted_q_relations * sorted_q_scores)
            ndcg_loss = self.ndcg_to_ndcg_loss_functoin(ndcg)
            losses.append(ndcg_loss)
        losses = torch.stack(losses)
        return self.reduction_method(losses)


@singleton
class RamezaniSuperSimpleLoss:
    def __init__(self, cosine_scaler=True):
        self.cosine_scaler = cosine_scaler
        self.log_instance()

    def log_instance(self):
        logger.info('Loss function: RamezaniSuperSimpleLoss')
        logger.info(f'RamezaniSuperSimpleLoss cosine_scaler: {self.cosine_scaler}')

    def calc(self, scores, relations_per_question):
        if self.cosine_scaler:
            scores = 0.5 * (1 + scores)
        loss = abs(relations_per_question - scores)
        loss = torch.sum(loss)
        return loss


@singleton
class BCELoss:
    def __init__(self, reduction='sum', cosine_scaler=True):
        check_list('reduction', reduction, ['mean', 'sum', 'none'])
        self.reduction = reduction
        self.cosine_scaler = cosine_scaler
        self.log_instance()

    def log_instance(self):
        logger.info('Loss function: BCELoss')
        logger.info(f'BCELoss reduction: {self.reduction}')
        logger.info(f'BCELoss cosine_scaler: {self.cosine_scaler}')

    def calc(self, scores, relations_per_question):
        if self.cosine_scaler:
            scores = 0.5 * (1 + scores)
        loss = F.binary_cross_entropy(scores, relations_per_question, reduction=self.reduction)
        return loss


@singleton
class RankCosineLoss:
    def __init__(self, reduction='sum', version='v1', cosine_scaler=True, v2_base_factor=10):
        check_list('reduction', reduction, ['mean', 'sum', 'none'])
        check_list('version', version, ['v1', 'v2'])
        self.reduction = reduction
        self.version = version
        self.cosine_scaler = cosine_scaler
        self.v2_base_factor = v2_base_factor
        self.reduction_method = _reduction_methods.get(self.reduction)
        similarity_to_loss_methods = {
            'v1': self.similarity_to_loss_v1,
            'v2': self.similarity_to_loss_v2,
        }
        self.similarity_to_loss_method = similarity_to_loss_methods.get(self.version)
        self.log_instance()

    def log_instance(self):
        logger.info('Loss function: RankCosineLoss')
        logger.info(f'RankCosineLoss reduction: {self.reduction}')
        logger.info(f'RankCosineLoss version: {self.version}')
        logger.info(f'RankCosineLoss cosine_scaler: {self.cosine_scaler}')
        logger.info(f'RankCosineLoss v2_base_factor: {self.v2_base_factor}')

    @staticmethod
    def similarity_to_loss_v1(similarity):
        '''Linear inverse scaling'''
        loss = 0.5 * (1 - similarity)
        return loss

    def similarity_to_loss_v2(self, similarity):
        '''Exponential inverse scaling'''
        loss = self.v2_base_factor ** (1 - similarity) - 1
        return loss

    def calc(self, scores, relations_per_question):
        if self.cosine_scaler:
            scores = 0.5 * (1 + scores)
        cosine_similarity = F.cosine_similarity(scores, relations_per_question)
        losses = self.similarity_to_loss_method(cosine_similarity)
        return self.reduction_method(losses)
