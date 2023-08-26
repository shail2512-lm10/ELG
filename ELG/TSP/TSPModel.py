import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TSP_Encoder, TSP_Decoder, _get_encoding
from models import reshape_by_heads, multi_head_attention


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        self.encoded_nodes = self.encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def one_step_rollout(self, state, cur_dist, cur_theta, scale, eval_type):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.tensor(random.sample(range(0, pomo_size), pomo_size), device=state.BATCH_IDX.device)[
                           None, :] \
                    .expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size), device=state.BATCH_IDX.device)

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, cur_dist=cur_dist, cur_theta=cur_theta, scale=scale, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            if eval_type == 'sample':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob
    
    
class E_TSP_Decoder(TSP_Decoder):

    def __init__(self, **model_params):
        super().__init__(**model_params)

        self.enable_EAS = None  # bool

        self.eas_W1 = None
        # shape: (batch, embedding, embedding)
        self.eas_b1 = None
        # shape: (batch, embedding)
        self.eas_W2 = None
        # shape: (batch, embedding, embedding)
        self.eas_b2 = None
        # shape: (batch, embedding)
        
    def init_eas_layers_random(self, batch_size):
        emb_dim = self.model_params['embedding_dim']  # 128
        init_lim = (1/emb_dim)**(1/2)
        
        weight1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((batch_size, emb_dim, emb_dim))
        bias1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((batch_size, emb_dim))
        self.eas_W1 = torch.nn.Parameter(weight1)
        self.eas_b1 = torch.nn.Parameter(bias1)
        self.eas_W2 = torch.nn.Parameter(torch.zeros(size=(batch_size, emb_dim, emb_dim)))
        self.eas_b2 = torch.nn.Parameter(torch.zeros(size=(batch_size, emb_dim)))

    def init_eas_layers_manual(self, W1, b1, W2, b2):
        self.eas_W1 = torch.nn.Parameter(W1)
        self.eas_b1 = torch.nn.Parameter(b1)
        self.eas_W2 = torch.nn.Parameter(W2)
        self.eas_b2 = torch.nn.Parameter(b2)

    def eas_parameters(self):
        return [self.eas_W1, self.eas_b1, self.eas_W2, self.eas_b2]
        
    def forward(self, encoded_last_node, ninf_mask, first_node=None, local_idx=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)
        # first_node.shape: (batch, modified_pomo)  # use first_node=None when pomo = {1, 2, ..., problem}

        head_num = self.model_params['head_num']
        batch_s = encoded_last_node.shape[0]
        n = encoded_last_node.shape[1]
        key_dim = self.k.shape[-1]
        emb_dim = encoded_last_node.shape[-1]
        input_s = ninf_mask.shape[2]
        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        if first_node is None:
            q_first = self.q_first
            # shape: (batch, head_num, pomo, qkv_dim)
        else:
            qkv_dim = self.model_params['qkv_dim']
            gathering_index = first_node[:, None, :, None].expand(-1, head_num, -1, qkv_dim)
            q_first = self.q_first.gather(dim=2, index=gathering_index)
            # shape: (batch, head_num, mod_pomo, qkv_dim)

        q = q_first + q_last

        if local_idx is not None:
            local_size = local_idx.shape[-1]
            local_k = torch.take_along_dim(self.k[:, :, None, :, :].expand(batch_s, head_num, n, input_s, key_dim), 
                                 local_idx[:, None, :, :, None].expand(batch_s, head_num, n, local_size, key_dim), dim=3)
            # shape: (batch, head_num, n, local, key_dim)
            local_v = torch.take_along_dim(self.v[:, :, None, :, :].expand(batch_s, head_num, n, input_s, key_dim), 
                                 local_idx[:, None, :, :, None].expand(batch_s, head_num, n, local_size, key_dim), dim=3)
            # shape: (batch, head_num, n, local, key_dim)
            local_single_head_key = torch.take_along_dim(self.single_head_key[:, None, :, :].expand(batch_s, n, emb_dim, input_s), 
                                 local_idx[:, :, :, None].expand(batch_s, n, local_size, emb_dim).transpose(3, 2), dim=3)
            # shape: (batch, n, local, key_dim)
            out_concat = multi_head_attention(q, local_k, local_v, rank3_ninf_mask=ninf_mask, local_idx=local_idx)
            # shape: (batch, pomo, head_num*qkv_dim)
        else:
            out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask, local_idx=local_idx)
            # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        # EAS Layer Insert
        #######################################################
        if self.enable_EAS:
            ms1 = torch.matmul(mh_atten_out, self.eas_W1)
            # shape: (batch, pomo, embedding)

            ms1 = ms1 + self.eas_b1[:, None, :]
            # shape: (batch, pomo, embedding)

            ms1_activated = F.relu(ms1)
            # shape: (batch, pomo, embedding)

            ms2 = torch.matmul(ms1_activated, self.eas_W2)
            # shape: (batch, pomo, embedding)

            ms2 = ms2 + self.eas_b2[:, None, :]
            # shape: (batch, pomo, embedding)
            
            mh_atten_out = mh_atten_out + ms2
            # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        if local_idx is not None:
            score = torch.matmul(mh_atten_out.unsqueeze(2), local_single_head_key).squeeze(2)
            # shape: (batch, pomo, local)
        else:
            score = torch.matmul(mh_atten_out, self.single_head_key)
            # shape: (batch, pomo, problem)
        
        sqrt_embedding_dim = self.model_params['embedding_dim'] ** 0.5
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem/local)

        score_clipped = logit_clipping * torch.tanh(score_scaled)
        
        if local_idx is not None:
            ninf_mask = torch.take_along_dim(ninf_mask, local_idx, dim=-1)
            
        score_masked = score_clipped + ninf_mask
        
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem/local)

        return probs