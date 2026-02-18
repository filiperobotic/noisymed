"""
Unified Training Script — dispatches to individual method implementations.

Supported methods:
    - baseline:           Standard CE + optional weighted loss
    - coteaching:         Co-teaching (Han et al., NeurIPS 2018)
    - gmm_filter:         GMM-based loss filtering
    - crass:              CRASS — Clinical Risk-Aware Sample Selection (Proposed)
    - dividemix:          DivideMix (Li et al., ICLR 2020)
    - unicon:             UNICON (Karim et al., CVPR 2022)
    - disc:               DISC (Li et al., CVPR 2023)
    - gmm_filter_crass:   GMM Filter + CRASS (Proposed)
    - dividemix_crass:    DivideMix + CRASS (Proposed)
    - disc_crass:         DISC + CRASS (Proposed)

    Cost-Sensitive variants:
    - baseline_cs:        Baseline + Cost-Sensitive Loss
    - coteaching_cs:      Co-teaching + Cost-Sensitive Loss
    - gmm_filter_cs:      GMM Filter + Cost-Sensitive Loss
    - dividemix_cs:       DivideMix + Cost-Sensitive Loss
    - unicon_cs:          UNICON + Cost-Sensitive Loss
    - disc_cs:            DISC + Cost-Sensitive Loss

Usage:
    python train.py --method baseline --dataset pneumoniamnist --noise_rate 0.2
    python train.py --method coteaching --dataset breastmnist --noise_rate 0.4
    python train.py --method crass --dataset pneumoniamnist --noise_rate 0.2 --lambda_risk 10
    python train.py --method dividemix --dataset dermamnist_bin --noise_rate 0.2 --weighted_loss
    python train.py --method gmm_filter_crass --dataset pneumoniamnist --noise_rate 0.2 --lambda_risk 20
    python train.py --method baseline_cs --dataset dermamnist_bin --noise_rate 0.2 --lambda_risk 20
    python train.py --method dividemix_cs --dataset dermamnist_bin --noise_rate 0.2 --lambda_risk 20
"""

import sys
import importlib

METHODS = {
    'baseline':         'train_baseline',
    'coteaching':       'train_coteaching',
    'gmm_filter':       'train_filter_loss',
    'crass':            'train_crass',
    'dividemix':        'train_dividemix',
    'unicon':           'train_unicon',
    'disc':             'train_disc',
    'gmm_filter_crass': 'train_gmm_filter_crass',
    'dividemix_crass':  'train_dividemix_crass',
    'disc_crass':       'train_disc_crass',
    # Cost-Sensitive variants
    'baseline_cs':      'train_baseline_cs',
    'coteaching_cs':    'train_coteaching_cs',
    'gmm_filter_cs':    'train_filter_loss_cs',
    'dividemix_cs':     'train_dividemix_cs',
    'unicon_cs':        'train_unicon_cs',
    'disc_cs':          'train_disc_cs',
}


def main():
    # Parse --method from argv before delegating to the method's own parser
    if '--method' not in sys.argv:
        print("Usage: python train.py --method <method_name> [method-specific args...]")
        print(f"\nAvailable methods: {', '.join(METHODS.keys())}")
        sys.exit(1)

    method_idx = sys.argv.index('--method')
    if method_idx + 1 >= len(sys.argv):
        print(f"Error: --method requires a value. Available: {', '.join(METHODS.keys())}")
        sys.exit(1)

    method = sys.argv[method_idx + 1]

    if method not in METHODS:
        print(f"Error: Unknown method '{method}'. "
              f"Available: {', '.join(METHODS.keys())}")
        sys.exit(1)

    # Remove --method <value> from argv so the method's argparse doesn't choke
    sys.argv = [sys.argv[0]] + sys.argv[:method_idx] + sys.argv[method_idx + 2:]
    # Remove duplicated sys.argv[0] if present
    if len(sys.argv) > 1 and sys.argv[0] == sys.argv[1]:
        sys.argv = sys.argv[1:]

    # Clean up: remove 'train.py' from argv[0] and keep only method args
    # Rebuild argv: script name + all remaining args
    remaining_args = []
    for i, arg in enumerate(sys.argv):
        if i == 0:
            remaining_args.append(arg)
        else:
            remaining_args.append(arg)
    sys.argv = remaining_args

    module_name = METHODS[method]
    print(f"\n{'='*60}")
    print(f"Dispatching to method: {method} ({module_name}.py)")
    print(f"{'='*60}\n")

    module = importlib.import_module(module_name)
    module.main()


if __name__ == '__main__':
    main()
