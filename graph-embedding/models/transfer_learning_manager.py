from .selector_model import SelectorModel
from database.utils import graph_data_to_database, raw_data_to_database, convert_to_prolog_data, convert_datatomap_to_prologdata


class TransferlearningManager:
    def __init__(self, base_path, source_domain_mapped_data):
        self._base_path = base_path

        self._source_domain_mapped_data = source_domain_mapped_data



        pass

    def kg_workflow(base_path, source_domain_name, mapping_key, source_domain_mapped_data, target_domain_name,
                    model_settings, use_recursion: bool, rounds: int = 3):
        target_domain_path = f"./preprocessed/{target_domain_name}"

        target_dataset = DatabaseWrapper()
        target_dataset.process_target(target_domain_path, target_domain_name)
        fold_splits = target_dataset.generate_fold_splits()

        out = []
        tl_not_possible = []

        for fold_configuration in fold_splits:
            for _ in range(rounds):
                result = {"transfer_learning": False, "fold_configuration": fold_configuration, "log": ""}
                # Generate dataset with misclassified by the model
                tl_dataset = DatabaseWrapper()

                if fold_configuration["train"][0] not in tl_not_possible:
                    knowledge_gap_data = generate_knowledge_gap_data(
                        dataset=target_dataset,
                        fold_configuration=fold_configuration,
                        use_recursion=use_recursion,
                        base_path=base_path
                    )

                    # Get transfer dataset
                    if knowledge_gap_data is not None:
                        transfer_learning_raw_dataset = get_transfer_dataset(
                            base_path=base_path,
                            source_domain_mapped_data=source_domain_mapped_data,
                            target_domain_name=target_domain_name,
                            target_domain_metadata=target_dataset.get_target_domain_metadata(),
                            knowledge_gap_data=knowledge_gap_data
                        )
                        if transfer_learning_raw_dataset is not None:
                            result["transfer_learning"] = True
                            tl_dataset.process_source_from_data(
                                {source_domain_name: {mapping_key: transfer_learning_raw_dataset}})
                        else:
                            tl_not_possible.append(fold_configuration["train"][0])
                            result["log"] += "Transfer-learning dataset could not be generated."
                    else:
                        tl_not_possible.append(fold_configuration["train"][0])
                        result["log"] += "Best model not found."

                tl_dataset.process_target(target_domain_path, target_domain_name)

                tl_train_metrics, tl_test_metrics = kg_training_workflow(
                    base_path=base_path,
                    dataset=tl_dataset,
                    use_recursion=use_recursion,
                    fold_configuration=fold_configuration,
                    model_settings=model_settings,
                )

                result["train"] = tl_train_metrics
                result["test"] = tl_test_metrics

                out.append(result)

        return out


    def get_transfer_dataset(base_path: str, source_domain_mapped_data, target_domain_name: str, target_domain_metadata, knowledge_gap_data):
        if knowledge_gap_data is not None:
            valid_predicates = list(source_domain_mapped_data["facts"].keys()) + list(source_domain_mapped_data["pos"].keys())

            selector_model = SelectorModel(valid_predicates, target=target_domain_name, target_domain_metadata=target_domain_metadata, prefix_path=base_path)

            # Learn the knowledge gap patterns
            selector_model.train(knowledge_gap_data)

            # Look for source domain samples whose patterns match knowledge gap ones.
            source_domain_converted_data = convert_datatomap_to_prologdata(source_domain_mapped_data)
            selector_model.test(source_domain_converted_data)

            pos_transfer_samples, neg_transfer_samples = selector_model.get_best_samples()

            raw_transfer_dataset = {
                "facts": convert_to_prolog_data(source_domain_mapped_data["facts"]),
                "pos": [f"{sample['raw']}." for sample in pos_transfer_samples],
                "neg": [f"{sample['raw'][1:]}." for sample in neg_transfer_samples]
            }

            selector_model.delete()

            return raw_transfer_dataset

        else:
            pass