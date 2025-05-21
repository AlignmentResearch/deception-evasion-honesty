import argparse
import os
import pickle as pkl
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import tqdm
import transformers
import wandb
from scipy.sparse import csr_array
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader  # type: ignore
from transformers import BitsAndBytesConfig
from trl import get_kbit_device_map

from solid_deception.detection.loaders import PromptDataset, collate_fn
from solid_deception.detection.residual import get_model_activations_parallel
from solid_deception.detection.sae import get_sae_features_parallel
from solid_deception.utils.training import handle_caching_from_config

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")


def pprint(x):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:  # type: ignore
        print(x)


def get_adaptive_logistic_regression(
    X_train, y_train, n_nonzero_coefs, max_iter=10_000, tol=1e-4, max_attempts=5
):
    l1_lambda_min, l1_lambda_max = 1e-5, 1e5
    for _ in range(max_attempts):
        l1_lambda = np.sqrt(l1_lambda_min * l1_lambda_max)
        lr = get_logistic_regression(
            X_train, y_train, penalty="l1", C=1 / l1_lambda, max_iter=max_iter
        )
        n_nonzero = np.sum(np.abs(lr.coef_[0]) > tol)

        if abs(n_nonzero - n_nonzero_coefs) <= 5:
            return lr
        elif n_nonzero > n_nonzero_coefs:
            l1_lambda_min = l1_lambda
        else:
            l1_lambda_max = l1_lambda
        pprint(f"number of nonzero coefficients: {n_nonzero}")

    return lr  # Return the last model if we don't converge # type: ignore


def activations_lists_to_train_test(true_activations, false_activations, test_frac, data_limit):

    # True is zero, 1 is false
    n_data = true_activations.shape[0]
    sklearn_labels = torch.concatenate((torch.zeros(n_data), torch.ones(n_data))).int()

    sklearn_data = torch.vstack((true_activations, false_activations)).float()
    idxs = np.random.permutation(len(sklearn_labels))
    sklearn_data = sklearn_data[idxs]
    sklearn_labels = sklearn_labels[idxs]
    if data_limit is not None:
        n_data = data_limit
        sklearn_data = sklearn_data[:data_limit]
        sklearn_labels = sklearn_labels[:data_limit]
    scaler_kwargs = {}

    n_train = int(n_data * (1 - test_frac))
    x_train = sklearn_data[:n_train]
    x_test = sklearn_data[n_train:]

    y_train = sklearn_labels[:n_train]
    y_test = sklearn_labels[n_train:]

    scaler = StandardScaler(**scaler_kwargs)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, scaler


def get_logistic_regression_cv(X_train, y_train, max_iter=10_000):
    # Fit the model
    t0 = time.time()
    model = LogisticRegressionCV(
        Cs=5,
        cv=3,
        max_iter=max_iter,
        penalty="elasticnet",
        solver="saga",
        verbose=False,
        l1_ratios=[0, 1],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Got logistic regression model!")
    print(f"took {time.time()-t0:.3f}s to fit LR")
    return model


def get_logistic_regression(X_train, y_train, penalty, C, max_iter=10_000):
    # Fit the model
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        penalty=penalty,
        solver="saga",
        verbose=False,
        warm_start=True,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def summarize_logistic_regression(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    # Get feature importances (coefficients)
    feature_importance = model.coef_[0]
    sorted_features = np.argsort(np.abs(feature_importance), kind="stable")[::-1]
    sorted_values = np.sort(np.abs(feature_importance), kind="stable")[::-1]
    zero_idxs = np.where(sorted_values == 0)
    if len(zero_idxs[0]) > 0:
        first_zero_idx = np.where(sorted_values == 0)[0][0]
    else:
        first_zero_idx = len(sorted_values)
    nonzero_features = sorted_features[:first_zero_idx]
    top_features = nonzero_features[:50]
    top_values = sorted_values[:first_zero_idx][:50]
    n_nonzero_features = len(nonzero_features)

    # Create summary dictionary
    if hasattr(model, "C_"):
        best_C = model.C_
    else:
        best_C = model.C
    if hasattr(model, "l1_ratio_"):
        best_l1_ratio = model.l1_ratio_
    else:
        best_l1_ratio = None
    summary = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "top_features": top_features,
        "top_feature_importance": feature_importance[top_features],
        "model": model,
        "best C_": best_C,
        "best l1_ratio": best_l1_ratio,
        "n_nonzero_features": n_nonzero_features,
        "nonzero_weights": top_values,
    }

    return summary


def get_activations_and_classify(
    args,
):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    config_dir = Path(args.csv_save_path).parent / Path("configs")
    metrics_to_save_path = str(config_dir / Path("detector"))

    if not args.no_cache:
        loaded_from_cache, (
            cached_csv_save_path,
            cached_lr_save_path,
            cached_metrics_to_save_path,
        ) = handle_caching_from_config(
            [args],
            [args.csv_save_path, args.lr_save_path, metrics_to_save_path],
            "detector",
            is_main_process,
        )
        if loaded_from_cache:
            return
    else:
        (cached_lr_save_path, cached_metrics_to_save_path) = None, None
        cached_csv_save_path = None

    if args.do_sae:
        print(f"SAE: {args.sae_path}, {args.sae_words_path}, {args.sae_descriptions_path}")
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "use_cache": False,  # No generation so we turn this off for now
        "attn_implementation": "flash_attention_2",
    }

    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model_kwargs.update(
            {
                "device_map": get_kbit_device_map(),  # type: ignore
                "quantization_config": quantization_config,  # type: ignore
            }
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, **model_kwargs
    )
    # if not quantize:
    #     model = model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

    df = pd.read_csv(args.data_path)
    train_df = deepcopy(df[df["split"] == "train"])
    train_lr_df = deepcopy(df[df["split"] == "train_lr"])
    test_df = deepcopy(df[df["split"] == "test"])
    # train_df = train_df.iloc[:200]

    # # DEBUG:
    if args.debug_training:
        train_df = train_df.iloc[:2048]
        train_lr_df = train_lr_df.iloc[:2048]
        test_df = test_df.iloc[:2048]
    accelerator = None

    # Get features from train_lr split to train the logistic regression
    if args.do_sae:
        train_lr_true_activations, train_lr_false_activations, accelerator = (
            get_sae_features_parallel(
                train_lr_df,
                model,
                tokenizer,
                args.sae_path,
                args.sae_words_path,
                args.sae_descriptions_path,
                batch_size=args.batch_size,
                layer=args.layer,
                max_length=args.max_length,
                top_k=args.top_k,
            )
        )
    else:
        train_lr_true_activations, train_lr_false_activations, accelerator = (
            get_model_activations_parallel(
                train_lr_df,
                model,
                tokenizer,
                batch_size=args.batch_size,
                layer=args.layer,
                max_length=args.max_length,
                all_positions=args.all_positions,
            )
        )

    # Get test set features for evaluation
    if args.do_sae:
        test_df_true_activations, test_df_false_activations, accelerator = (
            get_sae_features_parallel(
                test_df,
                model,
                tokenizer,
                args.sae_path,
                args.sae_words_path,
                args.sae_descriptions_path,
                batch_size=args.batch_size,
                accelerator=accelerator,
                layer=args.layer,
                max_length=args.max_length,
                top_k=args.top_k,
            )
        )
    else:
        test_df_true_activations, test_df_false_activations, accelerator = (
            get_model_activations_parallel(
                test_df,
                model,
                tokenizer,
                batch_size=args.batch_size,
                layer=args.layer,
                max_length=args.max_length,
                accelerator=accelerator,
                all_positions=args.all_positions,
            )
        )

    fraction_for_testing = 0.1 if not args.debug_training else 0.8
    X_train, X_test, y_train, y_test, scaler = activations_lists_to_train_test(
        train_lr_true_activations,
        train_lr_false_activations,
        fraction_for_testing,
        args.data_limit,
    )

    # This will do the fitting of the logistic regression separately on each process
    # if using `accelerate`. While wasteful of CPU, I can't work out a good way to
    # only do this on the main process and then sync up after.
    # We only save the result on the main process, so fact that the different logistic
    # regression parameters are different is not an issue

    if args.lr_path is None:
        print(f"Fitting layer {args.layer}")

        if not args.adaptive:
            lr = get_logistic_regression_cv(X_train, y_train, max_iter=10_000)
        else:
            lr = get_adaptive_logistic_regression(
                X_train, y_train, args.n_nonzero_coefs, max_iter=10_000
            )

    else:
        print(f"Loading logistic regression from {args.lr_path}")
        scaler, lr = pkl.load(open(args.lr_path, "rb"))

    summary = summarize_logistic_regression(lr, X_test, y_test)

    # Compute what the threshold would be for a specified fpr
    fprs, tprs, thresholds = roc_curve(y_test > 0.75, lr.predict_proba(X_test)[:, 1])

    if args.lie_false_positive_rate is not None:
        assert args.lie_true_positive_rate is None
        fpr_idx = np.where(fprs > args.lie_false_positive_rate)[0][0] - 1
        decision_boundary = thresholds[fpr_idx]
        actual_fpr, actual_tpr = fprs[fpr_idx], tprs[fpr_idx]

        if args.lie_false_positive_rate == 0.0:
            decision_boundary = 1.0

    else:
        assert args.lie_false_positive_rate is None
        tpr_idx = np.where(tprs > args.lie_true_positive_rate)[0][0]
        decision_boundary = thresholds[tpr_idx]
        actual_fpr, actual_tpr = fprs[tpr_idx], tprs[tpr_idx]

    pprint(summary)
    print(
        f"With FPR of {actual_fpr:.5f} and TPR of {actual_tpr:.5f}"
        f"threshold is {decision_boundary:.5f}"
    )
    # torch.distributed.barrier()

    if args.do_sae:
        train_df_true_activations, train_df_false_activations, _ = get_sae_features_parallel(
            train_df,
            model,
            tokenizer,
            args.sae_path,
            args.sae_words_path,
            args.sae_descriptions_path,
            batch_size=args.batch_size,
            accelerator=accelerator,
            layer=args.layer,
            max_length=args.max_length,
            top_k=args.top_k,
        )
    else:
        train_df_true_activations, train_df_false_activations, accelerator = (
            get_model_activations_parallel(
                train_df,
                model,
                tokenizer,
                batch_size=args.batch_size,
                layer=args.layer,
                max_length=args.max_length,
                accelerator=accelerator,
                all_positions=args.all_positions,
            )
        )
        print(f"Data size: {train_df_true_activations.shape}")
    false_data = train_df_false_activations.float()  # type: ignore
    true_data = train_df_true_activations.float()  # type: ignore

    print(f"Data size: {train_df_true_activations.shape}")
    # if is_main_process:
    false_data = scaler.transform(false_data)  # type: ignore
    true_data = scaler.transform(true_data)  # type: ignore

    train_df["deceptive_lr_lie_prob"] = lr.predict_proba(false_data)[:, 1]  # type: ignore
    train_df["truthful_lr_lie_prob"] = lr.predict_proba(true_data)[:, 1]  # type: ignore

    db = decision_boundary  # type: ignore
    train_df["lr_prediction_lie"] = train_df["deceptive_lr_lie_prob"] > db  # type: ignore
    train_df["lr_prediction_truth"] = train_df["truthful_lr_lie_prob"] > db  # type: ignore

    false_data = test_df_false_activations.float()  # type: ignore
    true_data = test_df_true_activations.float()  # type: ignore

    false_data = scaler.transform(false_data)
    true_data = scaler.transform(true_data)

    test_df["deceptive_lr_lie_prob"] = lr.predict_proba(false_data)[:, 1]  # type: ignore
    test_df["truthful_lr_lie_prob"] = lr.predict_proba(true_data)[:, 1]  # type: ignore
    test_df["lr_prediction_lie"] = test_df["deceptive_lr_lie_prob"] > db  # type: ignore
    test_df["lr_prediction_truth"] = test_df["truthful_lr_lie_prob"] > db  # type: ignore

    # Apply the trained logistic regression to the train_lr split for consistency
    train_lr_false_data = train_lr_false_activations.float()  # type: ignore
    train_lr_true_data = train_lr_true_activations.float()  # type: ignore

    train_lr_false_data = scaler.transform(train_lr_false_data)  # type: ignore
    train_lr_true_data = scaler.transform(train_lr_true_data)  # type: ignore

    deceptive_lr_lie_prob = lr.predict_proba(train_lr_false_data)[:, 1]  # type: ignore
    truthful_lr_lie_prob = lr.predict_proba(train_lr_true_data)[:, 1]  # type: ignore

    train_lr_df["deceptive_lr_lie_prob"] = deceptive_lr_lie_prob  # type: ignore
    train_lr_df["truthful_lr_lie_prob"] = truthful_lr_lie_prob  # type: ignore

    train_lr_df["lr_prediction_lie"] = train_lr_df["deceptive_lr_lie_prob"] > db  # type: ignore
    train_lr_df["lr_prediction_truth"] = train_lr_df["truthful_lr_lie_prob"] > db  # type: ignore

    if is_main_process:
        print(f"Average deceptive lr lie prob:{train_df['deceptive_lr_lie_prob'].mean():.3f}")
        print(f"Average truthful lr lie prob:{train_df['truthful_lr_lie_prob'].mean():.3f}")
        print(f"Average deceptive predicted as lie: {train_df['lr_prediction_lie'].mean():.3f}")

    # Include all three splits in the final dataframe
    df = pd.concat([train_df, train_lr_df, test_df])
    # Save only on the main process to avoid corruption
    if is_main_process:
        fprs, tprs, thresholds = roc_curve(y_test > 0.75, lr.predict_proba(X_test)[:, 1])
        wandb.init(  # type: ignore
            name=args.name,
        )

        wandb.config.update(  # type: ignore
            {
                "experiment_set_name": args.experiment_set_name,
                "experiment_type": "Detector",
                "data_limit": args.data_limit,
            }
        )
        metrics = {
            "specified_fpr": args.lie_false_positive_rate,
            "specified_tpr": args.lie_true_positive_rate,
            "fit_fpr": actual_fpr,
            "fit_tpr": actual_tpr,
            "threshold": decision_boundary,
            "full_dataset_tpr": (train_df["deceptive_lr_lie_prob"] > db).mean(),
            "full_dataset_fpr": (train_df["truthful_lr_lie_prob"] > db).mean(),
            "full_dataset_avg_deceptive_lr_lie": (train_df["deceptive_lr_lie_prob"]).mean(),
            "full_dataset_avg_truthful_lr_lie": (train_df["truthful_lr_lie_prob"]).mean(),
            "fit_roc": wandb.plot.roc_curve(y_test, lr.predict_proba(X_test)),  # type: ignore
            "fit_auc": summary["auc"],
            "fit_C": summary["best C_"],
        }

        wandb.log(metrics)  # type: ignore
        wandb.finish()  # type: ignore
        df.to_csv(args.csv_save_path, index=False)

        metrics_to_save = {"detector" + "/" + k: v for k, v in metrics.items() if k != "fit_roc"}
        config_dir = Path(args.csv_save_path).parent / Path("configs")
        os.makedirs(config_dir, exist_ok=True)

        if cached_csv_save_path is not None:
            df.to_csv(cached_csv_save_path, index=False)

        lr_learn_stats = (fprs, tprs, thresholds)
        pkl.dump(metrics_to_save, open(metrics_to_save_path, "wb"))
        if cached_metrics_to_save_path is not None:
            print(f"Dumped metrics to {cached_metrics_to_save_path}")
            pkl.dump(metrics_to_save, open(cached_metrics_to_save_path, "wb"))

        pkl.dump((scaler, lr, decision_boundary, lr_learn_stats), open(args.lr_save_path, "wb"))
        if cached_lr_save_path is not None:
            print(f"Dumped metrics to {cached_lr_save_path}")
            pkl.dump(
                (scaler, lr, decision_boundary, lr_learn_stats), open(cached_lr_save_path, "wb")
            )

    else:
        time.sleep(20)  # Allow main process to catch up before finishing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a residual detector for truthful vs deceptive responses"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pre-trained model"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default="logistic_regression",
        help="Name of run in wandb",
    )
    parser.add_argument(
        "--experiment_set_name",
        type=str,
        required=False,
        default="",
        help="Name of experiment set",
    )
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument(
        "--lr_path", type=str, required=False, help="Path to the logistic regression pickle"
    )
    parser.add_argument(
        "--csv_save_path", type=str, required=True, help="Path to the save location for the csv"
    )
    parser.add_argument(
        "--lr_save_path",
        type=str,
        required=True,
        help="Path to the save location for the logistic regression",
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=8, help="Batch size for inference"
    )
    parser.add_argument(
        "--layer", type=int, required=False, default=16, help="Layer to add probe at"
    )
    parser.add_argument(
        "--max_length", type=int, required=False, default=512, help="Max length of inputs"
    )
    parser.add_argument("--quantize", action="store_true", help="Whether to use 4-bit quantization")
    parser.add_argument("--do_sae", action="store_true", help="Whether to use an sae classifier")
    parser.add_argument(
        "--debug_training",
        action="store_true",
        help="Whether to limit the number of examples to speed up training",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Whether to be adaptive in the number of nonzero coefficients for the LR",
    )
    parser.add_argument(
        "--top_k",
        action="store_true",
        help="Whether to be use top-k cutoff in the SAE encoder",
    )

    parser.add_argument(
        "--n_nonzero_coefs",
        type=int,
        default=100,
        help="Desired number of nonzero coefficients",
    )
    parser.add_argument(
        "--lie_false_positive_rate",
        type=str,
        default=None,
        help="Level at which we set the lie detector false positive rate to"
        "choose a decision boundary",
    )
    parser.add_argument(
        "--lie_true_positive_rate",
        type=str,
        default=None,
        help="Level at which we set the lie detector true positive rate to "
        "choose a decision boundary. One of this and lie_false_positive_rate "
        " must be specified",
    )
    parser.add_argument(
        "--sae_path",
        type=str,
        help="Where to fetch the SAE from",
    )
    parser.add_argument(
        "--sae_words_path",
        type=str,
        help="Where to fetch the SAE wordslist from",
    )
    parser.add_argument(
        "--sae_descriptions_path",
        type=str,
        help="Where to fetch the SAE wordslist from",
    )
    parser.add_argument(
        "--data_limit",
        type=int,
        help="Limit to apply for number of verified input points",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random Seed",
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Do not run caching behavior (loading or saving)"
    )
    parser.add_argument(
        "--all_positions",
        action="store_true",
        help="Compute logistic regression features over all positions",
    )

    args = parser.parse_args()
    if args.lie_true_positive_rate == "None":
        args.lie_true_positive_rate = None
        args.lie_false_positive_rate = float(args.lie_false_positive_rate)
    if args.lie_false_positive_rate == "None":
        args.lie_false_positive_rate = None
        args.lie_true_positive_rate = float(args.lie_true_positive_rate)  # type: ignore
    get_activations_and_classify(
        args,
    )
