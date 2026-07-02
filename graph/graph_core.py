from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

EVENT_TYPES = [
    "PHENOMENON",
    "FAILURE",
    "ROOT_CAUSE",
    "ACTION",
    "VERIFICATION",
]
RELATION_TYPES = [
    "NONE",
    "CAUSE",
    "TEMPORAL",
    "TREAT",
    "VERIFY",
]

EVENT_TYPE_TO_ID = {name: idx for idx, name in enumerate(EVENT_TYPES)}
RELATION_TYPE_TO_ID = {name: idx for idx, name in enumerate(RELATION_TYPES)}
RELATION_ID_TO_TYPE = {idx: name for name, idx in RELATION_TYPE_TO_ID.items()}


class NodeSpanEncoder(nn.Module):
    """Aggregate span representations from multiple windows into document-level node vectors."""

    def __init__(self, hidden_size: int, category_count: int, dropout: float):
        super().__init__()
        self.type_embedding = nn.Embedding(len(EVENT_TYPES), 32)
        self.sent_embedding = nn.Embedding(128, 16)
        self.para_embedding = nn.Embedding(64, 16)
        self.order_embedding = nn.Embedding(64, 16)
        self.category_embedding = nn.Embedding(category_count, 16)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size * 3 + 32 + 16 + 16 + 16 + 16, hidden_size)

    def forward(
        self,
        sequence_output_list: List[List[torch.Tensor]],
        batch_nodes: List[List[Dict]],
        batch_windows: List[List[Dict]],
        batch_node_type_ids: List[torch.Tensor],
        batch_sentence_ids: List[torch.Tensor],
        batch_paragraph_ids: List[torch.Tensor],
        batch_node_order_ids: List[torch.Tensor],
        batch_category_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        device = batch_category_ids.device
        all_doc_node_vectors: List[torch.Tensor] = []

        for batch_idx, nodes in enumerate(batch_nodes):
            if sequence_output_list[batch_idx]:
                hidden_size = sequence_output_list[batch_idx][0].size(-1)
            else:
                hidden_size = self.proj.out_features

            if not nodes:
                all_doc_node_vectors.append(torch.zeros((0, hidden_size), device=device))
                continue

            node_vec_buckets = {node["node_id"]: [] for node in nodes}
            windows = batch_windows[batch_idx]
            seq_outputs = sequence_output_list[batch_idx]
            category_vec = self.category_embedding(batch_category_ids[batch_idx].to(device))

            for win_idx, window in enumerate(windows):
                seq_out = seq_outputs[win_idx]
                local_map = window["local_node_token_spans"]
                for node_local_idx, node in enumerate(nodes):
                    node_id = node["node_id"]
                    if node_id not in local_map:
                        continue

                    local_start, local_end = local_map[node_id]
                    start_vec = seq_out[local_start]
                    end_vec = seq_out[local_end]
                    span_mean = seq_out[local_start: local_end + 1].mean(dim=0)

                    type_vec = self.type_embedding(batch_node_type_ids[batch_idx][node_local_idx].to(device))
                    sent_vec = self.sent_embedding(batch_sentence_ids[batch_idx][node_local_idx].to(device))
                    para_vec = self.para_embedding(batch_paragraph_ids[batch_idx][node_local_idx].to(device))
                    order_idx = min(int(batch_node_order_ids[batch_idx][node_local_idx].item()), 63)
                    order_vec = self.order_embedding(torch.tensor(order_idx, device=device))

                    concat_vec = torch.cat(
                        [
                            start_vec,
                            end_vec,
                            span_mean,
                            type_vec,
                            sent_vec,
                            para_vec,
                            order_vec,
                            category_vec,
                        ],
                        dim=-1,
                    )
                    node_vec_buckets[node_id].append(self.proj(self.dropout(concat_vec)))

            node_vectors = []
            for node in nodes:
                bucket = node_vec_buckets[node["node_id"]]
                if not bucket:
                    bucket = [torch.zeros(hidden_size, device=device)]
                node_vectors.append(torch.stack(bucket, dim=0).mean(dim=0))
            all_doc_node_vectors.append(torch.stack(node_vectors, dim=0))

        return all_doc_node_vectors


class NodeInteractionLayer(nn.Module):
    """Apply lightweight global interaction among nodes in one document."""

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, node_vectors: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for vec in node_vectors:
            if vec.size(0) == 0:
                outputs.append(vec)
                continue
            outputs.append(self.encoder(vec.unsqueeze(0)).squeeze(0))
        return outputs


class PairRelationScorer(nn.Module):
    """Classify directed relation labels for every node pair."""

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.relpos_embedding = nn.Embedding(64, 16)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 4 + 16, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(RELATION_TYPES)),
        )

    def forward(self, node_vectors: torch.Tensor) -> torch.Tensor:
        node_count = node_vectors.size(0)
        if node_count == 0:
            return torch.zeros((0, 0, len(RELATION_TYPES)), device=node_vectors.device)

        logits = []
        for i in range(node_count):
            row_logits = []
            for j in range(node_count):
                rel_pos = min(abs(i - j), 63)
                rel_pos_vec = self.relpos_embedding(torch.tensor(rel_pos, device=node_vectors.device))
                pair_vec = torch.cat(
                    [
                        node_vectors[i],
                        node_vectors[j],
                        torch.abs(node_vectors[i] - node_vectors[j]),
                        node_vectors[i] * node_vectors[j],
                        rel_pos_vec,
                    ],
                    dim=-1,
                )
                row_logits.append(self.mlp(self.dropout(pair_vec)))
            logits.append(torch.stack(row_logits, dim=0))
        return torch.stack(logits, dim=0)


class ConstraintRegularizer(nn.Module):
    """Soft regularizer for direction and process consistency."""

    def __init__(self):
        super().__init__()

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.numel() == 0:
            return torch.tensor(0.0, device=probs.device)

        cause_idx = RELATION_TYPE_TO_ID["CAUSE"]
        temporal_idx = RELATION_TYPE_TO_ID["TEMPORAL"]
        verify_idx = RELATION_TYPE_TO_ID["VERIFY"]
        treat_idx = RELATION_TYPE_TO_ID["TREAT"]

        cause_prob = probs[:, :, cause_idx]
        temp_prob = probs[:, :, temporal_idx]
        verify_prob = probs[:, :, verify_idx]
        treat_prob = probs[:, :, treat_idx]

        anti_cause = (cause_prob * cause_prob.transpose(0, 1)).mean()
        anti_temp = (temp_prob * temp_prob.transpose(0, 1)).mean()
        verify_treat_balance = torch.relu(verify_prob.mean() - 2.0 * treat_prob.mean())
        return anti_cause + anti_temp + 0.2 * verify_treat_balance


class RelationGraphModel(nn.Module):
    """Relation graph model."""

    def __init__(self, pretrained_model_name: str, category_count: int, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name,
            local_files_only=True,
            use_safetensors=False,
        )
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.node_encoder = NodeSpanEncoder(hidden_size, category_count, dropout)
        self.node_interaction = NodeInteractionLayer(hidden_size, dropout)
        self.pair_scorer = PairRelationScorer(hidden_size, dropout)
        self.constraint_regularizer = ConstraintRegularizer()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_windows(self, batch_windows: List[List[Dict]]) -> List[List[torch.Tensor]]:
        all_outputs: List[List[torch.Tensor]] = []
        device = self.device

        for doc_windows in batch_windows:
            if not doc_windows:
                all_outputs.append([])
                continue

            input_ids = nn.utils.rnn.pad_sequence(
                [w["input_ids"] for w in doc_windows],
                batch_first=True,
                padding_value=0,
            ).to(device)
            attention_mask = nn.utils.rnn.pad_sequence(
                [w["attention_mask"] for w in doc_windows],
                batch_first=True,
                padding_value=0,
            ).to(device)
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq_output = self.dropout(encoder_outputs.last_hidden_state)
            all_outputs.append([seq_output[idx] for idx in range(seq_output.size(0))])
        return all_outputs

    def forward(self, batch: Dict) -> Dict:
        category_ids = batch["category_ids"].to(self.device)
        window_outputs = self.encode_windows(batch["windows"])
        node_vectors = self.node_encoder(
            sequence_output_list=window_outputs,
            batch_nodes=batch["nodes"],
            batch_windows=batch["windows"],
            batch_node_type_ids=batch["node_type_ids"],
            batch_sentence_ids=batch["sentence_ids"],
            batch_paragraph_ids=batch["paragraph_ids"],
            batch_node_order_ids=batch["node_order_ids"],
            batch_category_ids=category_ids,
        )
        node_vectors = self.node_interaction(node_vectors)
        pair_logits = [self.pair_scorer(doc_node_vec) for doc_node_vec in node_vectors]
        return {
            "node_vectors": node_vectors,
            "pair_logits": pair_logits,
        }


def apply_hard_constraints_and_decode(logits: torch.Tensor, role_mask: torch.Tensor) -> torch.Tensor:
    """Decode after masking role-incompatible relations."""
    if logits.numel() == 0:
        return torch.zeros((0, 0), dtype=torch.long, device=logits.device)
    masked_logits = logits.masked_fill(role_mask.to(logits.device) <= 0, -1e4)
    return masked_logits.argmax(dim=-1)


def extract_positive_edges(relation_matrix: torch.Tensor, ignore_label: int = -1) -> List[Tuple[int, int, str]]:
    """Convert a relation matrix to positive edges, skipping NONE and IGNORE."""
    edges: List[Tuple[int, int, str]] = []
    if relation_matrix.numel() == 0:
        return edges

    node_count = relation_matrix.size(0)
    none_id = RELATION_TYPE_TO_ID["NONE"]
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            rel_id = int(relation_matrix[i, j].item())
            if rel_id == ignore_label or rel_id < 0 or rel_id == none_id:
                continue
            if rel_id not in RELATION_ID_TO_TYPE:
                continue
            edges.append((i, j, RELATION_ID_TO_TYPE[rel_id]))
    return edges


def compute_violation_count(pred_relation_matrix: torch.Tensor, role_mask: torch.Tensor) -> int:
    """Count predicted edges that violate the role constraint mask."""
    node_count = pred_relation_matrix.size(0)
    violations = 0
    for i in range(node_count):
        for j in range(node_count):
            rel_id = int(pred_relation_matrix[i, j].item())
            if rel_id < 0:
                continue
            if role_mask[i, j, rel_id].item() <= 0:
                violations += 1
    return violations


__all__ = [
    "EVENT_TYPES",
    "RELATION_TYPES",
    "EVENT_TYPE_TO_ID",
    "RELATION_TYPE_TO_ID",
    "RELATION_ID_TO_TYPE",
    "NodeSpanEncoder",
    "NodeInteractionLayer",
    "PairRelationScorer",
    "ConstraintRegularizer",
    "RelationGraphModel",
    "apply_hard_constraints_and_decode",
    "extract_positive_edges",
    "compute_violation_count",
]
