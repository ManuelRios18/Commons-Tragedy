import importlib

SUBSTRATES = ["commons_harvest_uniandes"]


def get_game(substrate_name):

    if substrate_name not in SUBSTRATES:
        raise ValueError(f'{substrate_name} not in {SUBSTRATES}.')

    return importlib.import_module(f'{__name__}.{substrate_name}')
