import argparse

def process_folder(args, mode="af3"):
    import os
    import numpy as np
    from glob import glob
    from .utils import process_full_data_af3, process_full_data_boltz
    from .plotting import plot_paes, plot_plddt
    cwd = os.getcwd()
    accepted_modes = ["af3", "boltz2"]
    print(f"Processing folder {cwd} in {mode} mode")
    if mode not in accepted_modes:
        raise(ValueError(f"Accpeted modes are {' '.join(accepted_modes)}"))
        
    if mode == "af3":
        all_models = glob("full_data*json") 
        load_data = process_full_data_af3
        all_models = glob("*cif") 
    elif mode in ["af3", "boltz2"]:
        load_data = process_full_data_boltz
        all_models = glob("*cif") 
    
    print(f"There are {len(all_models)} {mode} models to process.")
    for struct_path in all_models:
        ## Load data
        model_data = load_data(struct_path)

        ## Plot PAE and PLDDT
        print(f"\tProcessing: {struct_path}")
        pae_name = struct_path.replace(".cif", "_pae.png")
        plddt_name = struct_path.replace(".cif", "_plddt.png")

        ptm = model_data["ptm"]
        iptm = model_data["iptm"]
        title_str = f"pTM: {ptm:.2f} | ipTM: {iptm:.2f}"
        plot_paes(model_data["pae"], chain_boundaries=model_data["chain_boundaries_by_res"], 
                  chain_ids=np.unique(model_data["token_chain_ids"]), title=title_str, save_name=pae_name)
        
        plot_plddt(model_data["res_plddts"], chain_boundaries=model_data["chain_boundaries_by_res"],
                   chain_ids=np.unique(model_data["token_chain_ids"]),
                   save_name=plddt_name)

def help(args):
    help_message = """
        How to use VirIn (VIRus INteractome) processor?
        List of available commands
        \t1. plot: generate PAE, plddt_plots  
    """
    print(help_message)

def main():
    parser = argparse.ArgumentParser(description="My Tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Register information of plotting PPIs
    parser_plot = subparsers.add_parser('plot', help='Do the first thing')
    parser_plot.add_argument("--mode", type=str, default="af3", help="Model used for the predicitons")
    parser_plot.set_defaults(func=process_folder)


    # Register help info
    parser_help = subparsers.add_parser('help', help='How to use the command line')
    parser_help.set_defaults(func=help)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()