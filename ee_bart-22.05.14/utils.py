from glob import glob
from pathlib import Path

import torch
from transformers import AutoConfig

from ee_bart.dataset import ACEDataset, ACEDatasetTokenBatcherAndLoader
from ee_bart.modeling_bart import ACEBartForConditionalGeneration
from ee_bart.tokenization_bart import ACEBartTokenizer
from ee_bart.linearization import ACETokens


def instantiate_model_and_tokenizer(
        pretrained_path,
        vocab_path,
        checkpoint=None,
        additional_tokens_smart_init=True,
        dropout=0.15,
        attention_dropout=0.15,
        from_pretrained=True,
        init_reverse=False,
        collapse_name_ops=False,
        use_pointer_tokens=True,
        dfs_linearization=True,
        bfs_linearization=True,
        use_entity_type=True,
        use_classification=True,
):
    tokenizer = ACEBartTokenizer.from_pretrained(
        pretrained_path,
        vocab_path,
        collapse_name_ops=collapse_name_ops,
        dfs_linearization=dfs_linearization,
        bfs_linearization=bfs_linearization,
        use_pointer_tokens=use_pointer_tokens,
        use_entity_type=use_entity_type,
        use_classification=use_classification
    )

    config = AutoConfig.from_pretrained(pretrained_path)
    config.output_attentions = True
    config.dropout = dropout
    config.stop_token_id = tokenizer.stop_token_id
    config.attention_dropout = attention_dropout
    if from_pretrained:
        model = ACEBartForConditionalGeneration.from_pretrained(pretrained_path, config=config)
    else:
        model = ACEBartForConditionalGeneration(config)

    model.resize_token_embeddings(len(tokenizer.encoder))

    if additional_tokens_smart_init:
        modified = 0
        for tok, idx in tokenizer.encoder.items():
            tok = tok.lstrip(tokenizer.INIT)

            if idx < tokenizer.old_enc_size:
                continue
            # 特殊pointer标记
            elif tok.startswith('<Event:') and tok.endswith('>'):
                tok_split = ['Event', str(tok.split(':')[1].strip('>'))]
            elif tok.startswith('<Argument:') and tok.endswith('>'):
                tok_split = ['Argument', str(tok.split(':')[1].strip('>'))]
            # 特殊标记
            elif tok.startswith('<'):
                continue
            # 事件类型节点
            elif ':' in tok:
                # 后序可能会有（“-”）的出现,需要处理一下。
                tok_split = tok.split(':')
                tok_split = tok_split[0].split('-') + tok_split[1].split('-')

            else:
                tok_split = tok.split('-')

            tok_split_ = tok_split
            tok_split = []
            for s in tok_split_:
                s_ = s + tokenizer.INIT
                if s_ in tokenizer.encoder:
                    tok_split.append(s_)
                else:
                    tok_split.extend(tokenizer._tok_bpe(s))

            vecs = []
            for s in tok_split:
                idx_split = tokenizer.encoder.get(s, -1)
                if idx_split > -1:
                    vec_split = model.model.shared.weight.data[idx_split].clone()
                    vecs.append(vec_split)

            if vecs:
                vec = torch.stack(vecs, 0).mean(0)
                noise = torch.empty_like(vec)
                noise.uniform_(-0.1, +0.1)
                model.model.shared.weight.data[idx] = vec + noise
                modified += 1

    if init_reverse:
        model.init_reverse_model()

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])

    return model, tokenizer


def instantiate_loader(
        path,
        tokenizer,
        batch_size=500,
        evaluation=True,
        use_recategorization=False,
        remove_longer_than=None,
        dereify=True,
):
    dataset = ACEDataset(
        path,
        tokenizer,
        use_recategorization=use_recategorization,
        remove_longer_than=remove_longer_than,
        dereify=dereify,
    )
    loader = ACEDatasetTokenBatcherAndLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not evaluation,
    )
    return loader
