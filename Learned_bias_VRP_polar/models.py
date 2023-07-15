import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        # self.global_encoder = CVRP_Global_Encoder(**model_params)

    def forward(self, depot_xy, node_xy_demand, dist):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)
        # dist.shape: (batch, problem+1, problem+1)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        
        embedded_node = self.embedding_node(node_xy_demand)
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        if self.model_params['encoder_mask'] is True:
            local_mask = float('-inf') * torch.ones(dist.shape, device=dist.device)
            _, local_idx_1 = dist.topk(self.model_params['local_size'], dim=-1, largest=False)
            _, local_idx_2 = dist.topk(self.model_params['local_size'], dim=-2, largest=False)
            src = torch.zeros(dist.shape, device=dist.device)
            local_mask = local_mask.scatter_(-1, local_idx_1, src)
            local_mask = local_mask.scatter_(-2, local_idx_2, src)
        else:
            local_mask = None

        for layer in self.layers:
            out = layer(out, local_mask)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1, mask):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v, rank3_ninf_mask=mask)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)

class EdgeNet(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        emb_dim = model_params['ff_hidden_dim']
        self.penalty = model_params['penalty']
        self.local_size = model_params['local_size'] + 1
        self.Linear_1 = nn.Linear(2 * self.local_size + 2, emb_dim)
        self.Linear_2 = nn.Linear(emb_dim, self.local_size)
    
    def zero_init(self):
        for _, param in self.named_parameters():
            nn.init.constant_(param, 0)
    
    def forward(self, theta, dist, ins_feature):
        # theta shape: (batch, multi, problem)
        # dist shape: (batch, multi, problem) 
        sorted_dist, idx = dist.topk(self.local_size, dim=-1, largest=False)
        # shape: (batch, multi, local)
        sorted_dist = sorted_dist / sorted_dist.max(-1)[0].unsqueeze(-1)
        sorted_theta = torch.take_along_dim(theta, idx, dim=-1)

        # padding (0, 0) for selected nodes
        # sorted_theta[sorted_dist == float('inf')] = 0
        # sorted_dist[sorted_dist == float('inf')] = 0
        
        sorted_dist_theta = torch.cat((sorted_dist, sorted_theta), dim=-1)
        x_in = torch.cat((sorted_dist_theta, ins_feature[0]), dim=-1)
        x_in = torch.cat((x_in, ins_feature[1]), dim=-1)
        # shape: (batch, multi, 2 * local + 2)
        edge_emb = self.Linear_1(x_in)
        # shape: (batch, multi, emb)
        out = self.Linear_2(edge_emb) - sorted_dist
        # shape: (batch, multi, local)
        out_mat = self.penalty * torch.ones(dist.shape, device=dist.device)
        # shape: (batch, multi, problem+1)
        out = out_mat.scatter_(-1, idx, out)
        # shape: (batch, multi, problem+1)

        return out


class EdgeNet_(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        emb_dim = int(model_params['ff_hidden_dim'] / 2)
        self.penalty = model_params['penalty']
        self.local_size = model_params['local_size'] + 1
        self.local_Linear = nn.Linear(2, emb_dim)
        self.global_Linear = nn.Linear(emb_dim + 4, 1)
    
    def zero_init(self):
        for _, param in self.named_parameters():
            nn.init.constant_(param, 0)
    
    def forward(self, theta, dist, ins_feature):
        # theta shape: (batch, multi, problem)
        # dist shape: (batch, multi, problem)
        sorted_dist, idx = dist.topk(self.local_size, dim=-1, largest=False)
        # shape: (batch, multi, local)

        sorted_theta = torch.take_along_dim(theta, idx, dim=-1).unsqueeze(-1)
        # shape: (batch, multi, local, 1)
        sorted_theta_dist = torch.cat((sorted_theta, sorted_dist.unsqueeze(-1)), dim=-1)
        # shape: (batch, multi, local, 2)
        ins_feature = torch.cat((ins_feature[0], ins_feature[1]), dim=-1)[:, :, None, :].expand(sorted_theta_dist.shape)
        local_emb = self.local_Linear(sorted_theta_dist)
        # shape: (batch, multi, local, emb)
        pooling_global_emb = local_emb.mean(-2)
        # shape: (batch, multi, emb)
        
        local_global_emb = torch.cat((sorted_theta_dist, pooling_global_emb[:, :, None, :].expand(local_emb.shape)), dim=-1)
        # shape: (batch, multi, local, emb + 2)
        local_global_emb = torch.cat((local_global_emb, ins_feature), dim=-1)
        # shape: (batch, multi, local, emb + 4)

        out = self.global_Linear(local_global_emb).squeeze(-1) - sorted_dist
        # shape: (batch, multi, local)
        out_mat = self.penalty * torch.ones(dist.shape, device=dist.device)
        # shape: (batch, multi, problem+1)
        out = out_mat.scatter_(-1, idx, out)
        # shape: (batch, multi, problem+1)

        return out


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim+1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        if self.model_params['bias'] == 'learn':
            if self.model_params['set_mlp']:
                self.edge_net = EdgeNet_(self.model_params)
            else:
                self.edge_net = EdgeNet(self.model_params)
            self.edge_net.zero_init()

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, load, cur_dist, cur_theta, ins_feature, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['embedding_dim'] ** 0.5
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        if self.model_params['bias'] == 'learn':
            edge_affinity = self.edge_net(cur_theta, cur_dist, ins_feature)
            score_scaled += edge_affinity
            # shape: (batch, pomo, problem)
        elif self.model_params['bias'] == 'dis':
            penalty_mat = - 2 ** 0.5 * self.model_params['penalty'] * torch.ones(cur_dist.shape, device=cur_dist.device)
            sorted_dist, idx = cur_dist.topk(self.model_params['local_size'], dim=-1, largest=False)
            cur_dist = penalty_mat.scatter_(-1, idx, sorted_dist)
            score_scaled -= cur_dist / 2 ** 0.5
            # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs
    

########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    if len(qkv.shape) == 4:
        qkv = qkv.reshape(qkv.size(0) * qkv.size(1), qkv.size(2), qkv.size(3))
        # shape: (batch * multi, n, head_num*key_dim)
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)
    
    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding) or (batch, multi, problem, embeddin)

        embedding_dim = input1.shape[-1]
        norm_dim = 1.
        for shape in input1.shape[:-1]:
            norm_dim = norm_dim * shape
        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(norm_dim, embedding_dim))
        back_trans = normalized.reshape(input1.shape)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))