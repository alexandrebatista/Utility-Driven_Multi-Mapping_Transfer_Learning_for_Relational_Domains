from mapping.mapping import generate_mappings
from mapping.apply_mappings import apply_mappings

if __name__ == '__main__':
    # Exemplo de diretórios: 'y' para origem e 'x' para destino

    source_path = './preprocessed/imdb'

    source_directory = './preprocessed/imdb/fold00'
    target_directory = './preprocessed/cora/fold00'
    solutions = generate_mappings(source_directory, target_directory)

    solutions.print_coherent_mappings()

    #output_path = "/home/alexandre/mestrado/graph-embedding/mapped_datasets/to_cora/imdb"

    #process_dataset(source_path, output_path, solutions.get_useful_mappings())
    #apply_mappings(source_path, output_path, solutions.get_useful_mappings())
