import logging
from typing import Optional

from srlearn import Database


def load_database(
    path: str = None,
    use_recursion: bool = False,
    target_predicate: str = None,
    reset_target_predicate: bool = False,
    neg_pos_ratio = 1,
    max_failed_neg_sampling_retries: Optional[int] = 50,
    logger: logging.Logger = None
):
    if not logger:
        logger = logging.getLogger("Database loader")

    database = Database().fromFiles(
        facts = f"{path}/facts.pl",
        pos = f"{path}/pos.pl",
        neg = f"{path}/neg.pl",
        modes = f"{path}/modes.pl",
        useRecursion = use_recursion
    )
    logger.debug(f"Database loaded with success.")

    if reset_target_predicate:
        logger.warn(f"`resetTargetPredicate` is set to True. Target predicate `{target_predicate}` will be ignored and target relation will be set to `None`. Set `resetTargetPredicate` to False if it is desired to set `targetPredicate` as the target relation.")
        database = database.resetTargetPredicate()
    else:
        if target_predicate:
            database = database.setTargetPredicate(
                target_predicate,
                useRecursion = use_recursion,
                negPosRatio = neg_pos_ratio,
                maxFailedNegSamplingRetries = max_failed_neg_sampling_retries
            )
        else:
            default_target_predicate = database.getTargetRelation()
            if default_target_predicate:
                database.neg = database.generateNegativePredicates(
                    negPosRatio = neg_pos_ratio,
                    maxFailedSamplingRetries = max_failed_neg_sampling_retries,
                    keepCurrentNegativeExamples = True
                )
                logger.warn(f"No target predicate was given. It will be set as `{default_target_predicate}` by default. Set `targetPredicate` if it is desired to set another relation as the target relation.")
            else:
                logger.warn(f"No target predicate was given. It will be set as `None` by default. Set `targetPredicate` if it is desired to set any relation as the target relation.")

    return database
