# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from .bin_tokens import WeightedCE, resolve_new_token_ids
from .budget import (DurationHead, gather_marker_states, LastHiddenCapture,
                     DUR_TOK, DUR_END)
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        # --- budget-aware AD -------------------------------------------------
        self.budget_variant = getattr(finetuning_args, "budget_variant", "none")
        self.dur_head, self.marker_id, self._dur_log = None, None, {}
        if self.budget_variant != "none":
            self.marker_id = getattr(self.processing_class, "budget_marker_id", None)
            cfg = self.model.config
            hidden = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
            dtype = getattr(self.model, "dtype", torch.bfloat16)
            self.dur_head = DurationHead(hidden).to(device=self.args.device, dtype=dtype)
            logger.info_rank0(
                f"budget variant {self.budget_variant}: duration head on hidden={hidden}, "
                f"marker id={self.marker_id}, lambda={finetuning_args.budget_lambda}")
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self._wce = None
        self._wce_cap = None
        self._bin_log = {}
        self._wce_ids = getattr(finetuning_args, "_special_token_ids", None)
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self, *args, **kwargs) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer(*args, **kwargs)

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        elif self._wce_ids:
            labels = inputs.pop("labels")
            inputs.pop("dur_targets", None)
            # Hook the final norm rather than output_hidden_states=True: the latter
            # returns all 37 layers and accelerate then upcasts every one to fp32,
            # which is a 15 GB allocation at this sequence length.
            # Under DDP/FSDP the model is wrapped and does not forward
            # get_output_embeddings(); hook and read weights off the real module.
            base = getattr(model, "module", model)
            if self._wce_cap is None:
                self._wce_cap = LastHiddenCapture(base)
            # logits_to_keep=1 stops the model building the full [1, T, 248320]
            # logit tensor (8.6 GB bf16 at 17k tokens, which accelerate then
            # upcasts to fp32). The hidden states we need come from the hook, and
            # the weighted CE below computes its own logits in fused chunks.
            model(**inputs, logits_to_keep=1)
            hidden = self._wce_cap.h
            lm_w = base.get_output_embeddings().weight
            if self._wce is None:
                self._wce = WeightedCE(self._wce_ids,
                                       self.finetuning_args.special_token_loss_weight)
            loss = self._wce(hidden, lm_w, labels)
            self._bin_log = dict(self._wce.last)
            return loss
        elif self.dur_head is not None:
            targets = inputs.pop("dur_targets", None)
            outputs = model(**inputs, output_hidden_states=True)
            loss = outputs.loss
            self._dur_log = {"ce": float(loss.detach())}
            if targets is not None and targets.numel():
                states = gather_marker_states(outputs.hidden_states[-1], inputs["input_ids"], self.marker_id)
                n = min(states.size(0), targets.size(0))
                if n:
                    pred = self.dur_head(states[:n])
                    tgt = targets[:n].to(pred.device)
                    clamp = getattr(self.finetuning_args, "budget_clamp", 0.0)
                    pf, tf = pred.float(), tgt.float()
                    if clamp:
                        pf = pf.clamp(-clamp / 6.0, clamp)
                        tf = tf.clamp(-clamp / 6.0, clamp)
                    mse = torch.nn.functional.mse_loss(pf, tf)
                    # head computes in fp32; cast back so backward sees one dtype
                    loss = loss + self.finetuning_args.budget_lambda * mse.to(loss.dtype)
                    self._dur_log.update(mse=float(mse.detach()),
                                         mae=float((pf - tf).abs().mean().detach()),
                                         n=int(n))
            return loss
        else:
            return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def create_optimizer(self):
        opt = super().create_optimizer()
        if self.dur_head is not None and not getattr(self, "_head_in_optim", False):
            opt.add_param_group({"params": list(self.dur_head.parameters()),
                                 "lr": self.args.learning_rate})
            self._head_in_optim = True
            logger.info_rank0("duration head added to the optimizer")
        return opt

    @override
    def log(self, logs, *args, **kwargs):
        if self._dur_log:
            logs.update({f"dur/{k}": v for k, v in self._dur_log.items()})
        if self._bin_log:
            logs.update({f"bin/{k}": v for k, v in self._bin_log.items()})
        return super().log(logs, *args, **kwargs)

    @override
    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.dur_head is not None:
            self.dur_head.save(output_dir or self.args.output_dir)


    @torch.no_grad()
    def _stream_rows(self, kind, rows):
        """Append this step's rows to a per-rank file and flush to disk."""
        if not rows:
            return
        rank, world = self.args.process_index, self.args.world_size
        steps = getattr(self, "_stream_steps", {})
        step = steps.get(kind, 0)
        steps[kind] = step + 1
        self._stream_steps = steps
        n = len(rows)
        d = os.path.join(self.args.output_dir, "stream")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{kind}.rank{rank}.jsonl"), "a", encoding="utf-8") as f:
            for j, r in enumerate(rows):
                r = {"idx": step * world * n + rank * n + j, **r}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _merge_stream(self, kind, out_name, n_total=None):
        """Merge per-rank stream files into one file in dataset order (rank 0 only)."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if not self.is_world_process_zero():
            return
        import glob
        rows = {}
        for p in glob.glob(os.path.join(self.args.output_dir, "stream", f"{kind}.rank*.jsonl")):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                rows.setdefault(r["idx"], r)  # padded duplicate batches repeat an idx-ordered sample
        keys = sorted(k for k in rows if n_total is None or k < n_total)
        path = os.path.join(self.args.output_dir, out_name)
        with open(path, "w", encoding="utf-8") as f:
            for k in keys:
                f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
        logger.info_rank0(f"Merged {len(keys)} streamed {kind} rows into {path}")

    def _teacher_forced_step(self, model, inputs, labels):
        """Force every token except the body of each audio description.

        The prompt, every dialogue line, each timestamp and the `M:SS.s [AD] `
        prefix are taken from ground truth; only what follows the prefix is
        generated, up to the newline. This isolates description quality from
        timestamp prediction, so results compare directly with models that are
        handed ground-truth AD timings.
        """
        tok = self.processing_class
        ad_prefix_re = re.compile(r"^(\d+:\d\d\.\d \[AD\] )")
        nl = tok("\n", add_special_tokens=False)["input_ids"]
        eos = tok.eos_token_id
        max_ad = int(getattr(self.finetuning_args, "budget_max_ad_tokens", 160))
        cap = LastHiddenCapture(model) if self.dur_head is not None else None
        outs = []
        for b in range(inputs["input_ids"].size(0)):
            lab = labels[b]
            keep = lab[lab != IGNORE_INDEX]
            # keep <|dur|>/<|dur_end|>: the target is re-tokenized below to force
            # the dialogue lines, and stripping them would change what the model sees
            target = tok.decode(keep, skip_special_tokens=False)
            target = target.replace(tok.eos_token or "", "").replace(tok.pad_token or "", "")
            prompt_ids = inputs["input_ids"][b][lab == IGNORE_INDEX].tolist()
            # Build the KV cache once over the prompt, then extend it. Rebuilding
            # the whole prefix for every audio description costs an O(n^2) attention
            # pass each time - 27 GB on a 30k-token clip, and far slower besides.
            produced, dur_log = [], []
            past, dev = None, self.args.device

            chunk = int(getattr(self.finetuning_args, "budget_prefill_chunk", 2048))

            def feed(tok_ids):
                """Extend the cache with these tokens; return the last model output.

                Long prompts are pushed through in chunks. A single forward over a
                30k-token prompt needs ~28 GB of attention workspace; feeding it in
                2k blocks keeps that bounded because each block only attends to the
                cache, never recomputing the full square.
                """
                nonlocal past
                out = None
                for i in range(0, len(tok_ids), chunk):
                    blk = tok_ids[i:i + chunk]
                    out = model(input_ids=torch.tensor([blk], device=dev),
                                past_key_values=past, use_cache=True,
                                output_hidden_states=(self.dur_head is not None
                                                      and not (cap is not None and cap.ok)))
                    past = out.past_key_values
                if os.environ.get("BUDGET_DEBUG"):
                    try:
                        L = past.get_seq_length() if hasattr(past, "get_seq_length") else past[0][0].shape[-2]
                    except Exception:
                        L = "?"
                    logger.info_rank0(f"[tf] fed {len(tok_ids)} tok -> cache len {L}")
                return out

            feed(prompt_ids)
            for line in target.split("\n"):
                m = ad_prefix_re.match(line)
                if not m:                                    # dialogue line: forced
                    feed(tok(line, add_special_tokens=False)["input_ids"] + nl)
                    produced.append(line)
                    continue
                # force the "M:SS.s [AD] " prefix; its logits give the first AD token
                out = feed(tok(m.group(1), add_special_tokens=False)["input_ids"])
                gen = []
                for _ in range(max_ad):
                    nxt = int(out.logits[0, -1].argmax())
                    if nxt == eos or "\n" in tok.decode([nxt]):
                        break
                    gen.append(nxt)
                    out = feed([nxt])
                    if self.dur_head is not None and nxt == self.marker_id:
                        h = cap.h[0, -1:] if (cap is not None and cap.ok) \
                            else out.hidden_states[-1][0, -1:]
                        dur_log.append(round(float(self.dur_head(h)[0]), 3))
                feed(nl)
                produced.append(m.group(1) + tok.decode(gen, skip_special_tokens=False))
            del past
            torch.cuda.empty_cache()
            raw = "\n".join(produced)
            clean = raw
            for _t in (DUR_TOK, DUR_END):
                clean = clean.replace(_t, "")
            clean = re.sub(r"[ \t]{2,}", " ", clean)
            outs.append({"predict": clean, "predict_raw": raw, "label": target,
                         "duration_head": dur_log})
        if cap is not None:
            cap.close()
        self._tf_outputs = getattr(self, "_tf_outputs", []) + outs
        self._stream_rows("teacher_forced", outs)
        return outs

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if getattr(self.finetuning_args, "budget_teacher_forced", False):
            labels = inputs.get("labels")
            self._teacher_forced_step(model, inputs, labels)
            return None, None, labels

        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()
            tok = self.processing_class
            strip = {t for t in (tok.pad_token, tok.eos_token, "<|endoftext|>", "<|im_end|>") if t}
            rows = []
            for b in range(generated_tokens.size(0)):
                gen = generated_tokens[b, inputs["input_ids"].size(-1):].tolist()
                text = tok.decode(gen, skip_special_tokens=False)
                for t in strip:
                    text = text.replace(t, "")
                lab = labels[b] if labels is not None else None
                lab_text = ""
                if lab is not None:
                    lab_text = tok.decode(lab[lab != IGNORE_INDEX].tolist(), skip_special_tokens=False)
                    for t in strip:
                        lab_text = lab_text.replace(t, "")
                rows.append({"predict": text, "label": lab_text})
            self._stream_rows("generated", rows)

        return loss, generated_tokens, labels

    def save_teacher_forced(self, output_dir=None, n_total=None) -> None:
        """Merge the per-rank teacher-forced stream files into one ordered file.

        Every rank holds only its own shard of _tf_outputs, so writing them all to
        one path (the old behaviour) made the ranks overwrite each other.
        """
        self._merge_stream("teacher_forced", "teacher_forced_predictions.jsonl", n_total)

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
