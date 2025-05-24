from mapping.main import mappings_workflow
from models.main import models_workflow


if __name__ == '__main__':
    import os

    os.nice(10)

    # ------------------------------------------------------------------------------------------------------------------
    # Generating Mappings and saving files

    #mappings_workflow()

    # ------------------------------------------------------------------------------------------------------------------
    # Transforming the datasets in Srlearn databases

    models_workflow(
        target_domain="nell_finances",
        train_folds=["fold00"],
        source_domains=["nell_sports"],
        quality="best"
    )

    models_workflow(
        target_domain="nell_finances",
        train_folds=["fold00"],
        source_domains=["nell_sports"],
        quality="worst"
    )
