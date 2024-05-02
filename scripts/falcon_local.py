r"""
    Evaluate NDT2 decoder on the Falcon Challenge.

    H1: https://wandb.ai/joelye9/context_general_bci/runs/edf4h5ym
    M1: https://wandb.ai/joelye9/context_general_bci/runs/93snpffp
"""

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from context_general_bci.falcon_decoder import NDT2Decoder

def run_evaluate(
        evaluation='local',
        model_path='./local_data/ndt2_h1_sample.pth',
        config_stem='falcon/h1/h1',
        zscore_path='./local_data/ndt2_zscore_h1.pt',
        split='h1',
        phase='minival'
    ):
    
    evaluator = FalconEvaluator(
        eval_remote=evaluation == "remote",
        split=split)

    task = getattr(FalconTask, split)
    config = FalconConfig(task=task)

    decoder = NDT2Decoder(
        task_config=config,
        model_ckpt_path=model_path,
        model_cfg_stem=config_stem,
        zscore_path=zscore_path,
        dataset_handles=[x.stem for x in evaluator.get_eval_files(phase=phase)],
        batch_size=1
    )


    return evaluator.evaluate(decoder, phase=phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", "-e", type=str, default='local', choices=["local", "remote"], 
    )
    parser.add_argument(
        "--model-path", "-m", type=str, default='./local_data/ndt2_h1_sample.pth'
    )
    parser.add_argument(
        "--config-stem", "-c", type=str, default='falcon/h1/h1',
        help="Name in context-general-bci codebase for config. \
            Currently, directly referencing e.g. a local yaml is not implemented unclear how to get Hydra to find it in search path."
    )
    parser.add_argument(
        "--zscore-path", "-z", type=str, default='./local_data/ndt2_zscore_h1.pt'
    )
    parser.add_argument(
        '--split', "-s", type=str, choices=['h1', 'h2', 'm1', 'm2'], default='h1',
    )
    parser.add_argument(
        '--phase', "-p", choices=['minival', 'test'], default='minival'
    )
    args = parser.parse_args()

    run_evaluate(
        evaluation=args.evaluation,
        model_path=args.model_path,
        config_stem=args.config_stem,
        zscore_path=args.zscore_path,
        split=args.split,
        phase=args.phase
    )