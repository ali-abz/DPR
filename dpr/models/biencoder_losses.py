import torch
import torch.nn.functional as F
from torch import Tensor as T
# from IPython import embed

import logging

EPS = 1e-12


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
    def __init__(self, reduction='mean', version='v1', ndcg_op='div', min_shift=False, cosine_scaler=False, min_max_scaler=False, devider_eps_shift=EPS, show_details_every=100):
        check_list('reduction', reduction, ['mean', 'sum', 'none'])
        check_list('version', version, ['v1', 'v2'])
        check_list('ndcg_op', ndcg_op, ['divide', 'difference'])
        self.reduction = reduction
        self.version = version
        self.min_shift = min_shift
        self.cosine_scaler = cosine_scaler
        self.min_max_scaler = min_max_scaler
        self.devider_eps_shift = devider_eps_shift
        self.show_details_every = show_details_every
        self.detail_counter = 0
        self.reduction_method = _reduction_methods.get(self.reduction)
        ndcg_to_ndcg_loss_methods = {
            'v1': self.ndcg_to_ndcg_loss_v1,
            'v2': self.ndcg_to_ndcg_loss_v2,
        }
        self.ndcg_op = ndcg_op
        self.ndcg_op_function = self.get_ndcg_op()
        self.ndcg_to_ndcg_loss_functoin = ndcg_to_ndcg_loss_methods.get(self.version)
        self.log_instance()

    def get_ndcg_op(self):
        if self.ndcg_op == 'divide':
            return lambda larger_num, smaller_num: smaller_num / larger_num
        elif self.ndcg_op == 'difference':
            return lambda larger_num, smaller_num: larger_num - smaller_num
        else:
            raise ValueError(f'Did not expect {self.ndcg_op} as ndcg_op')
    # TODO: check if we need v1 or v2 when using difference TODO TODO
    def log_instance(self):
        logger.info('Loss function: A3GLoss')
        logger.info(f'A3GLoss reduction: {self.reduction}')
        logger.info(f'A3GLoss version: {self.version}')
        logger.info(f'A3GLoss ndcg_op: {self.ndcg_op}')
        logger.info(f'A3GLoss cosine_scaler: {self.cosine_scaler}')
        logger.info(f'A3GLoss min_max_scaler: {self.min_max_scaler}')
        logger.info(f'A3GLoss devider_eps_shift: {self.devider_eps_shift}')

    def _NDCG(self, ranks: T):
        positions = torch.stack([torch.log2(x) for x in torch.arange(2., len(ranks) + 2)])
        positions = positions.to(ranks.device) + self.devider_eps_shift
        if self.min_shift:
            min_value = abs(min(ranks.min().detach(), 0))
            ranks += min_value
        dcg = (ranks / positions).sum()
        sorted_ranks = torch.stack(sorted(ranks, reverse=True))
        ideal_dcg = (sorted_ranks / positions).sum() + self.devider_eps_shift
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

    def show_details(self, original_scores, scores, relations, losses, loss):
        self.detail_counter += 1
        if self.detail_counter % self.show_details_every == 0:
            logger.info(f'BATCH DETAILS: original_scores: {original_scores}')
            logger.info(f'BATCH DETAILS: scaled_scores: {scores}')
            logger.info(f'BATCH DETAILS: relations: {relations}')
            logger.info(f'BATCH DETAILS: losses: {losses}')
            logger.info(f'BATCH DETAILS: loss: {loss}')
            self.log_instance()

    def calc(self, scores: T, relations: T):
        if self.cosine_scaler:
            scaled_scores = 0.5 * (1 + scores)
        if self.min_max_scaler:
            scaled_scores = scores - scores.min(1, keepdim=True)[0]
            scaled_scores = scaled_scores / scaled_scores.max(1, keepdim=True)[0]
        losses = []
        for q_scores, q_relations in zip(scaled_scores, relations):
            sorted_q_scores, sorted_q_relations = zip(*sorted(zip(q_scores, q_relations), reverse=True))
            sorted_q_scores = torch.stack(sorted_q_scores)
            sorted_q_relations = torch.stack(sorted_q_relations)
            ndcg = self._NDCG(sorted_q_relations * sorted_q_scores + EPS)
            ndcg_loss = self.ndcg_to_ndcg_loss_functoin(ndcg)
            losses.append(ndcg_loss)
        losses = torch.stack(losses)
        loss = self.reduction_method(losses)
        self.show_details(scores, scaled_scores, relations, losses, loss)
        return loss


@singleton
class RamezaniSuperSimpleLoss:
    def __init__(self, cosine_scaler=True, min_max_scaler=False, reduction='sum', show_details_every=100):
        self.cosine_scaler = cosine_scaler
        self.min_max_scaler = min_max_scaler
        if min_max_scaler == cosine_scaler:
            raise ValueError(f'min_max_scaler and cosine_scaler are both: {min_max_scaler}')
        self.reduction = reduction
        self.reduction_method = _reduction_methods.get(self.reduction)
        self.show_details_every = show_details_every
        self.detail_counter = 0
        self.log_instance()

    def log_instance(self):
        logger.info('Loss function: RamezaniSuperSimpleLoss')
        logger.info(f'RamezaniSuperSimpleLoss cosine_scaler: {self.cosine_scaler}')
        logger.info(f'RamezaniSuperSimpleLoss min_max_scaler: {self.min_max_scaler}')
        logger.info(f'RamezaniSuperSimpleLoss reduction: {self.reduction}')

    def show_details(self, original_scores, scores, relations, losses, loss):
        self.detail_counter += 1
        if self.detail_counter % self.show_details_every == 0:
            logger.info(f'BATCH DETAILS: original_scores: {original_scores}')
            logger.info(f'BATCH DETAILS: scaled_scores: {scores}')
            logger.info(f'BATCH DETAILS: relations: {relations}')
            logger.info(f'BATCH DETAILS: losses: {losses}')
            logger.info(f'BATCH DETAILS: loss: {loss}')
            self.log_instance()

    def calc(self, scores, relations):
        if self.cosine_scaler:
            scaled_scores = 0.5 * (1 + scores)
        if self.min_max_scaler:
            scaled_scores = scores - scores.min(1, keepdim=True)[0]
            scaled_scores = scaled_scores / scaled_scores.max(1, keepdim=True)[0]
        losses = abs(relations - scaled_scores)
        loss = self.reduction_method(losses)
        self.show_details(scores, scaled_scores, relations, losses, loss)
        return loss


@singleton
class BCELoss:
    def __init__(self, reduction='sum', cosine_scaler=True, min_max_scaler=False, show_details_every=100):
        check_list('reduction', reduction, ['mean', 'sum', 'none'])
        self.reduction = reduction
        self.reduction_method = _reduction_methods.get(self.reduction)
        self.cosine_scaler = cosine_scaler
        self.min_max_scaler = min_max_scaler
        if min_max_scaler == cosine_scaler:
            raise ValueError(f'min_max_scaler and cosine_scaler are both: {min_max_scaler}')
        self.show_details_every = show_details_every
        self.detail_counter = 0
        self.log_instance()

    def log_instance(self):
        logger.info('Loss function: BCELoss')
        logger.info(f'BCELoss reduction: {self.reduction}')
        logger.info(f'BCELoss cosine_scaler: {self.cosine_scaler}')
        logger.info(f'BCELoss min_max_scaler: {self.min_max_scaler}')

    def show_details(self, original_scores, scores, relations, losses, loss):
        self.detail_counter += 1
        if self.detail_counter % self.show_details_every == 0:
            logger.info(f'BATCH DETAILS: original_scores: {original_scores}')
            logger.info(f'BATCH DETAILS: scaled_scores: {scores}')
            logger.info(f'BATCH DETAILS: relations: {relations}')
            logger.info(f'BATCH DETAILS: losses: {losses}')
            logger.info(f'BATCH DETAILS: loss: {loss}')
            self.log_instance()

    def calc(self, scores, relations):
        if self.cosine_scaler:
            scaled_scores = 0.5 * (1 + scores)
        if self.min_max_scaler:
            scaled_scores = scores - scores.min(1, keepdim=True)[0]
            scaled_scores = scaled_scores / scaled_scores.max(1, keepdim=True)[0]
        losses = F.binary_cross_entropy(scaled_scores, relations, reduction='none')
        loss = self.reduction_method(losses)
        self.show_details(scores, scaled_scores, relations, losses, loss)
        return loss


@singleton
class RankCosineLoss:
    def __init__(self, reduction='sum', version='v1', cosine_scaler=True, min_max_scaler=False, v2_base_factor=10, show_details_every=100):
        check_list('reduction', reduction, ['mean', 'sum', 'none'])
        check_list('version', version, ['v1', 'v2'])
        self.reduction = reduction
        self.version = version
        self.cosine_scaler = cosine_scaler
        self.min_max_scaler = min_max_scaler
        if min_max_scaler == cosine_scaler:
            raise ValueError(f'min_max_scaler and cosine_scaler are both: {min_max_scaler}')
        self.v2_base_factor = v2_base_factor
        self.show_details_every = show_details_every
        self.detail_counter = 0
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
        logger.info(f'RankCosineLoss min_max_scaler: {self.min_max_scaler}')
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

    def show_details(self, original_scores, scores, relations, losses, loss):
        self.detail_counter += 1
        if self.detail_counter % self.show_details_every == 0:
            logger.info(f'BATCH DETAILS: original_scores: {original_scores}')
            logger.info(f'BATCH DETAILS: scaled_scores: {scores}')
            logger.info(f'BATCH DETAILS: relations: {relations}')
            logger.info(f'BATCH DETAILS: losses: {losses}')
            logger.info(f'BATCH DETAILS: loss: {loss}')
            self.log_instance()

    def calc(self, scores: T, relations):
        if self.cosine_scaler:
            scaled_scores = 0.5 * (1 + scores)
        if self.min_max_scaler:
            scaled_scores = scores - scores.min(1, keepdim=True)[0]
            scaled_scores = scaled_scores / scaled_scores.max(1, keepdim=True)[0]
        cosine_similarity = F.cosine_similarity(scaled_scores, relations)
        losses = self.similarity_to_loss_method(cosine_similarity)
        loss = self.reduction_method(losses)
        self.show_details(scores, scaled_scores, relations, losses, loss)
        return loss


@singleton
class KLDivergence:
    def __init__(self, cosine_scaler=False, min_max_scaler=True, reduction='sum', show_details_every=100):
        self.cosine_scaler = cosine_scaler
        self.min_max_scaler = min_max_scaler
        if min_max_scaler == cosine_scaler:
            raise ValueError(f'min_max_scaler and cosine_scaler are both: {min_max_scaler}')
        self.reduction = reduction
        self.reduction_method = _reduction_methods.get(self.reduction)
        self.show_details_every = show_details_every
        self.detail_counter = 0
        self.log_instance()

    def log_instance(self):
        logger.info('Loss function: KLDivergence')
        logger.info(f'KLDivergence cosine_scaler: {self.cosine_scaler}')
        logger.info(f'KLDivergence min_max_scaler: {self.min_max_scaler}')
        logger.info(f'KLDivergence reduction: {self.reduction}')

    def show_details(self, original_scores, scores, relations, losses, loss):
        self.detail_counter += 1
        if self.detail_counter % self.show_details_every == 0:
            logger.info(f'BATCH DETAILS: original_scores: {original_scores}')
            logger.info(f'BATCH DETAILS: scaled_scores: {scores}')
            logger.info(f'BATCH DETAILS: relations: {relations}')
            logger.info(f'BATCH DETAILS: losses: {losses}')
            logger.info(f'BATCH DETAILS: loss: {loss}')
            self.log_instance()

    def calc(self, scores, relations):
        if self.cosine_scaler:
            scaled_scores = 0.5 * (1 + scores)
        if self.min_max_scaler:
            scaled_scores = scores - scores.min(1, keepdim=True)[0]
            scaled_scores = scaled_scores / scaled_scores.max(1, keepdim=True)[0]
        losses = scaled_scores * torch.log(scaled_scores / (relations + EPS) + 1)
        loss = self.reduction_method(losses)
        self.show_details(scores, scaled_scores, relations, losses, loss)
        return loss
