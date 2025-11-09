from typing import Optional, Tuple
import os

import numpy as np
import yaml
import argparse
import shutil
import textwrap
import time
from copy import deepcopy

from threadpoolctl import threadpool_limits

from dingo.core.posterior_models.build_model import (
    autocomplete_model_kwargs,
    build_model_from_kwargs,
)
from dingo.gw.training.train_builders import (
    build_dataset,
    set_train_transforms,
    build_svd_for_embedding_network,
)
from dingo.core.utils.trainutils import RuntimeLimits
from dingo.core.utils import (
    set_requires_grad_flag,
    get_number_of_model_parameters,
    build_train_and_test_loaders,
)
from dingo.core.utils.trainutils import EarlyStopping
from dingo.gw.dataset import WaveformDataset
from dingo.core.posterior_models import BasePosteriorModel


def copy_files_to_local(
    file_path: str, local_dir: Optional[str], leave_keys_on_disk: bool, is_condor: bool = False,
) -> str:
    """
    Copy files to local node if local_dir is provided to minimize network traffic during training.
    """
    local_file_path = file_path
    if local_dir is not None:
        file_name = file_path.split("/")[-1]
        local_file_path = os.path.join(local_dir, file_name)
        print(f"Copying file to {local_file_path}")
        # Copy file
        start_time = time.time()
        shutil.copy(file_path, local_file_path)
        elapsed_time = time.time() - start_time
        print("Done. This took {:2.0f}:{:2.0f} min.".format(*divmod(elapsed_time, 60)))
    elif leave_keys_on_disk and is_condor:
        print(
            f"Warning: leave_waveforms_on_disk defaults to True, but local_cache_path is not specified. "
            f"This means that the waveforms will be loaded during training from {local_file_path} ."
            f"This can lead to unexpected long times for data loading during training due to network traffic. "
            f"To prevent this, specify 'local_cache_path = tmp' in the local settings or set "
            f"leave_waveforms_on_disk = False. However, the latter is not recommended for large datasets since "
            f"it can lead to memory issues when loading the entire dataset into RAM. "
        )

    return local_file_path


def _resolve_waveform_model(
    explicit: Optional[str],
    data_settings: dict,
    fallback_meta: Optional[str] = None,
) -> str:
    """
    Resolve the waveform model from (in order): explicit arg, data_settings, checkpoint metadata.
    """
    if explicit:
        return explicit
    if "waveform_model" in data_settings and data_settings["waveform_model"]:
        return data_settings["waveform_model"]
    if fallback_meta:
        return fallback_meta
    raise ValueError(
        "Waveform model not specified. Provide via --waveform_model or data.waveform_model in the settings, "
        "or ensure it is present in the checkpoint metadata."
    )


def prepare_training_new(
    train_settings: dict, train_dir: str, local_settings: dict, waveform_model: Optional[str] = None
) -> Tuple[BasePosteriorModel, WaveformDataset]:
    """
    Initialize WaveformDataset and PosteriorModel for a NEW run, with an explicit waveform model.
    """
    data_settings = deepcopy(train_settings["data"])

    # Resolve and record waveform model
    resolved_model = _resolve_waveform_model(waveform_model, data_settings)
    data_settings["waveform_model"] = resolved_model  # ensure present downstream

    # Optionally copy files to local and update path
    data_settings["waveform_dataset_path"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset_path"],
        local_dir=local_settings.get("local_cache_path", None),
        leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        is_condor=True if "condor" in local_settings else False,
    )

    # Build dataset (builders can read waveform_model from data_settings or kwargs)
    wfd = build_dataset(
        data_settings=data_settings,
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        # Optional: some implementations accept this explicitly
        waveform_model=resolved_model,
    )
    initial_weights = {}

    # Optional SVD seeding for embedding network — computed for THIS waveform model
    if train_settings["model"].get("embedding_kwargs", None):
        print("\nBuilding SVD for initialization of embedding network.")
        initial_weights["V_rb_list"] = build_svd_for_embedding_network(
            wfd,
            train_settings["data"],
            train_settings["training"]["stage_0"]["asd_dataset_path"],
            num_workers=local_settings["num_workers"],
            batch_size=train_settings["training"]["stage_0"]["batch_size"],
            out_dir=train_dir,
            waveform_model=resolved_model,  # NEW
            **train_settings["model"]["embedding_kwargs"]["svd"],
        )

    # Set initial transforms (model-aware)
    set_train_transforms(
        wfd,
        train_settings["data"],
        train_settings["training"]["stage_0"]["asd_dataset_path"],
        waveform_model=resolved_model,  # NEW
    )

    # Configure model from a representative sample of THIS dataset/model
    autocomplete_model_kwargs(train_settings["model"], wfd[0])

    full_settings = {
        "dataset_settings": wfd.settings,
        "train_settings": train_settings,
        "waveform_model": resolved_model,  # persist in metadata for resume
    }

    print("\nInitializing new posterior model.")
    print("Complete settings:")
    print(yaml.dump(full_settings, default_flow_style=False, sort_keys=False))

    pm = build_model_from_kwargs(
        settings=full_settings,
        initial_weights=initial_weights,
        device=local_settings["device"],
    )

    if local_settings.get("wandb", False):
        try:
            import wandb

            wandb.init(
                config=full_settings,
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            print("WandB is enabled but not installed.")

    return pm, wfd


def prepare_training_resume(
    checkpoint_name: str, local_settings: dict, train_dir: str, waveform_model: Optional[str] = None
) -> Tuple[BasePosteriorModel, WaveformDataset]:
    """
    Load PosteriorModel and WaveformDataset from a checkpoint (resume) and ensure waveform model is resolved.
    """
    pm = build_model_from_kwargs(
        filename=checkpoint_name, device=local_settings["device"]
    )

    data_settings = deepcopy(pm.metadata["train_settings"]["data"])

    # Resolve waveform model (CLI > settings > checkpoint metadata)
    resolved_model = _resolve_waveform_model(
        waveform_model, data_settings, fallback_meta=pm.metadata.get("waveform_model")
    )
    data_settings["waveform_model"] = resolved_model

    # Optionally copy files to local and update path
    data_settings["waveform_dataset_path"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset_path"],
        local_dir=local_settings.get("local_cache_path", None),
        leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        is_condor=True if "condor" in local_settings else False,
    )

    wfd = build_dataset(
        data_settings=data_settings,
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        waveform_model=resolved_model,  # NEW
    )

    if local_settings.get("wandb", False):
        try:
            import wandb

            wandb.init(
                resume="must",
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            print("WandB is enabled but not installed.")

    # Ensure the model metadata contains waveform_model for future resumes
    if "waveform_model" not in pm.metadata or pm.metadata["waveform_model"] != resolved_model:
        if hasattr(pm, "metadata"):
            pm.metadata["waveform_model"] = resolved_model

    return pm, wfd


def initialize_stage(
    pm: BasePosteriorModel,
    wfd: WaveformDataset,
    stage: dict,
    num_workers: int,
    resume: bool = False,
):
    """
    Initialize stage: transforms, loaders, (optionally) optimizer/scheduler, and RB layer freeze, model-aware.
    """
    train_settings = pm.metadata["train_settings"]
    waveform_model = pm.metadata.get("waveform_model") or train_settings["data"].get("waveform_model")
    if not waveform_model:
        raise ValueError("Missing 'waveform_model' in pm.metadata or train_settings['data'].")

    # Rebuild transforms based on current noise AND waveform model.
    set_train_transforms(
        wfd,
        train_settings["data"],
        stage["asd_dataset_path"],
        waveform_model=waveform_model,  # NEW
    )

    # Allows for changes in batch size between stages.
    train_loader, test_loader = build_train_and_test_loaders(
        wfd,
        train_settings["data"]["train_fraction"],
        stage["batch_size"],
        num_workers,
    )

    if not resume:
        # New optimizer and scheduler. If we are resuming, these should have been loaded from the checkpoint.
        print("Initializing new optimizer and scheduler.")
        pm.optimizer_kwargs = stage["optimizer"]
        pm.scheduler_kwargs = stage["scheduler"]
        pm.initialize_optimizer_and_scheduler()

    # Freeze/unfreeze RB layer if necessary
    if "freeze_rb_layer" in stage:
        if stage["freeze_rb_layer"]:
            set_requires_grad_flag(
                pm.network, name_contains="layers_rb", requires_grad=False
            )
        else:
            set_requires_grad_flag(
                pm.network, name_contains="layers_rb", requires_grad=True
            )
    n_grad = get_number_of_model_parameters(pm.network, (True,))
    n_nograd = get_number_of_model_parameters(pm.network, (False,))
    print(f"Fixed parameters: {n_nograd}\nLearnable parameters: {n_grad}\n")

    return train_loader, test_loader


def train_stages(
    pm: BasePosteriorModel, wfd: WaveformDataset, train_dir: str, local_settings: dict
) -> bool:
    """
    Train the network, iterating through the sequence of stages.
    """
    train_settings = pm.metadata["train_settings"]
    runtime_limits = RuntimeLimits(
        epoch_start=pm.epoch, **local_settings["runtime_limits"]
    )

    # Extract list of stages from settings dict
    stages = []
    num_stages = 0
    while True:
        try:
            stages.append(train_settings["training"][f"stage_{num_stages}"])
            num_stages += 1
        except KeyError:
            break
    end_epochs = list(np.cumsum([stage["epochs"] for stage in stages]))

    num_starting_stage = np.searchsorted(end_epochs, pm.epoch + 1)
    for n in range(num_starting_stage, num_stages):
        stage = stages[n]

        if pm.epoch == end_epochs[n] - stage["epochs"]:
            print(f"\nBeginning training stage {n}. Settings:")
            print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader = initialize_stage(
                pm, wfd, stage, local_settings["num_workers"], resume=False
            )
        else:
            print(f"\nResuming training in stage {n}. Settings:")
            print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader = initialize_stage(
                pm, wfd, stage, local_settings["num_workers"], resume=True
            )
        early_stopping = None
        if stage.get("early_stopping"):
            try:
                early_stopping = EarlyStopping(**stage["early_stopping"])
            except Exception:
                print(
                    "Early stopping settings invalid. Please pass 'patience', 'delta', 'metric'"
                )
                raise

        runtime_limits.max_epochs_total = end_epochs[n]
        pm.train(
            train_loader,
            test_loader,
            train_dir=train_dir,
            runtime_limits=runtime_limits,
            checkpoint_epochs=local_settings["checkpoint_epochs"],
            use_wandb=local_settings.get("wandb", False),
            test_only=local_settings.get("test_only", False),
            early_stopping=early_stopping,
        )
        # if test_only, model should not be saved, and run is complete
        if local_settings.get("test_only", False):
            return True

        if pm.epoch == end_epochs[n]:
            save_file = os.path.join(train_dir, f"model_stage_{n}.pt")
            print(f"Training stage complete. Saving to {save_file}.")
            pm.save_model(save_file, save_training_info=True)
        if runtime_limits.local_limits_exceeded(pm.epoch):
            print("Local runtime limits reached. Ending program.")
            break

    if pm.epoch == end_epochs[-1]:
        return True
    else:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Train a neural network for gravitational-wave single-event inference.

        This program can be called in one of two ways:
            a) with a settings file. This will create a new network based on the 
            contents of the settings file.
            b) with a checkpoint file. This will resume training from the checkpoint.
        """
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        help="YAML file containing training settings.",
    )
    parser.add_argument(
        "--train_dir", required=True, help="Directory for Dingo training output."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint file from which to resume training.",
    )
    parser.add_argument(
        "--exit_command",
        type=str,
        default="",
        help="Optional command to execute after completion of training.",
    )
    parser.add_argument(
        "--waveform_model",
        type=str,
        default=None,
        help="Identifier/name of the waveform model used to generate the dataset (e.g. IMRPhenomXPHM).",
    )
    args = parser.parse_args()

    # The settings file and checkpoint are mutually exclusive.
    if args.checkpoint is None and args.settings_file is None:
        parser.error("Must specify either a checkpoint file or a settings file.")
    if args.checkpoint is not None and args.settings_file is not None:
        parser.error("Cannot specify both a checkpoint file and a settings file.")

    return args


def train_local():
    args = parse_args()

    os.makedirs(args.train_dir, exist_ok=True)

    if args.settings_file is not None:
        print("Beginning new training run.")
        with open(args.settings_file, "r") as fp:
            train_settings = yaml.safe_load(fp)

        # Extract the local settings from train settings file, save it separately.
        local_settings = train_settings.pop("local")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "w") as f:
            if (
                local_settings.get("wandb", False)
                and "id" not in local_settings["wandb"].keys()
            ):
                try:
                    import wandb

                    local_settings["wandb"]["id"] = wandb.util.generate_id()
                except ImportError:
                    print("wandb not installed, cannot generate run id.")
            yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

        pm, wfd = prepare_training_new(
            train_settings, args.train_dir, local_settings, waveform_model=args.waveform_model
        )

    else:
        print("Resuming training run.")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
            local_settings = yaml.safe_load(f)
        pm, wfd = prepare_training_resume(
            args.checkpoint, local_settings, args.train_dir, waveform_model=args.waveform_model
        )

    with threadpool_limits(limits=1, user_api="blas"):
        complete = train_stages(pm, wfd, args.train_dir, local_settings)

    if complete:
        if args.exit_command:
            print(
                f"All training stages complete. Executing exit command: {args.exit_command}."
            )
            os.system(args.exit_command)
        else:
            print("All training stages complete.")
    else:
        print("Program terminated due to runtime limit.")
