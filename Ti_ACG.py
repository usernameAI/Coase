import math
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax


class Disentangled_Multi_Head_Attention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(Disentangled_Multi_Head_Attention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout_list = nn.ModuleList(nn.Dropout(attn_dropout_prob) for _ in range(2))
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.ln_list = nn.ModuleList(nn.LayerNorm(normalized_shape=hidden_size, eps=1e-12) for _ in range(6))
        self.linear_dim2head_list = nn.ModuleList(nn.Linear(hidden_size, self.all_head_size) for _ in range(7))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask, absolute_pos_Q, absolute_pos_K, time_matrix_emb_Q, time_matrix_emb_K):
        mixed_input_tensor = self.ln_list[0](input_tensor)
        mixed_absolute_pos_Q = self.ln_list[1](absolute_pos_Q)
        mixed_absolute_pos_K = self.ln_list[2](absolute_pos_K)
        mixed_time_matrix_emb_Q = self.ln_list[3](time_matrix_emb_Q)
        mixed_time_matrix_emb_K = self.ln_list[4](time_matrix_emb_K)
        mixed_query_layer = self.linear_dim2head_list[0](mixed_input_tensor)
        mixed_key_layer = self.linear_dim2head_list[1](mixed_input_tensor)
        mixed_value_layer = self.linear_dim2head_list[2](mixed_input_tensor)
        mixed_absolute_pos_Q = self.linear_dim2head_list[3](mixed_absolute_pos_Q)
        mixed_absolute_pos_K = self.linear_dim2head_list[4](mixed_absolute_pos_K)
        mixed_time_matrix_emb_Q = self.linear_dim2head_list[5](mixed_time_matrix_emb_Q)
        mixed_time_matrix_emb_K = self.linear_dim2head_list[6](mixed_time_matrix_emb_K)
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        absolute_pos_Q_ = self.transpose_for_scores(mixed_absolute_pos_Q).permute(0, 2, 1, 3)
        time_matrix_emb_Q_ = self.transpose_for_scores(mixed_time_matrix_emb_Q).permute(0, 3, 1, 2, 4)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        absolute_pos_K_ = self.transpose_for_scores(mixed_absolute_pos_K).permute(0, 2, 3, 1)
        time_matrix_emb_K_ = self.transpose_for_scores(mixed_time_matrix_emb_K).permute(0, 3, 1, 2, 4)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)  # [100, 1, 20, 100]
        attn_score1 = torch.matmul(query_layer, key_layer)
        attn_score2 = torch.matmul(query_layer, absolute_pos_K_)
        attn_score3 = torch.matmul(time_matrix_emb_K_, query_layer.unsqueeze(-1)).squeeze(-1)
        attn_score4 = torch.mean(torch.matmul(time_matrix_emb_Q_, time_matrix_emb_K_.permute(0, 1, 2, 4, 3)), dim=-1)
        attn_score5 = torch.matmul(time_matrix_emb_Q_, key_layer.permute(0, 1, 3, 2).unsqueeze(-1)).squeeze(-1)
        attn_score6 = torch.matmul(time_matrix_emb_Q_, absolute_pos_K_.permute(0, 1, 3, 2).unsqueeze(-1)).squeeze(-1)
        attn_score7 = torch.matmul(time_matrix_emb_K_, absolute_pos_Q_.unsqueeze(-1)).squeeze(-1)
        attn_score8 = torch.matmul(absolute_pos_Q_, absolute_pos_K_)
        attn_score9 = torch.matmul(absolute_pos_Q_, key_layer)
        attention_probs = attn_score1 + attn_score2 + attn_score3 + attn_score4 + attn_score5 + attn_score6 + attn_score7 + attn_score8 + attn_score9
        attention_probs = attention_probs / self.sqrt_attention_head_size
        attention_probs = attention_probs + attention_mask
        attention_probs = self.softmax(attention_probs)
        attention_probs = self.attn_dropout_list[0](attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.ln_list[5](context_layer)
        context_layer = self.out_dropout(context_layer)
        hidden_states = self.dense(context_layer) + input_tensor
        return hidden_states


class SudokuFormer(nn.Module):
    def __init__(self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, n_layers, layer_norm_eps=1e-12):
        super(SudokuFormer, self).__init__()
        self.n_layers = n_layers
        self.multi_head_attention_list = nn.ModuleList(Disentangled_Multi_Head_Attention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        ) for _ in range(self.n_layers))
        self.mlp_list = nn.ModuleList(MLP_ln_trans(input_dim=hidden_size, embed_dim=intermediate_size,
                                                   output_dim=hidden_size, dropout=hidden_dropout_prob)
                                      for _ in range(self.n_layers))

    def forward(self, hidden_states, attention_mask, absolute_pos_Q, absolute_pos_K, time_matrix_emb_Q,
                time_matrix_emb_K):
        feature_list = []
        for layer in range(self.n_layers):
            hidden_states = self.multi_head_attention_list[layer](hidden_states, attention_mask, absolute_pos_Q,
                                                                  absolute_pos_K, time_matrix_emb_Q, time_matrix_emb_K)
            hidden_states = self.mlp_list[layer](hidden_states)
            feature_list.append(hidden_states)
        hidden_states = sum(feature_list) / len(feature_list)
        return hidden_states


class MLP_bn(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, dropout):
        super(MLP_bn, self).__init__()
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(input_dim)
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = self.bn(x)
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.output_projection(x)
        return x


class MLP_ln(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, dropout):
        super(MLP_ln, self).__init__()
        self.dropout = dropout
        self.ln = nn.LayerNorm(normalized_shape=input_dim, eps=1e-12)
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.ln1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-12)

    def forward(self, x):
        x = self.ln(x)
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.ln1(x)
        x = self.dropout(x)
        x = self.output_projection(x)
        return x


class MLP_ln_trans(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, dropout):
        super(MLP_ln_trans, self).__init__()
        self.dropout = dropout
        self.ln = nn.LayerNorm(normalized_shape=input_dim, eps=1e-12)
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.ln1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-12)

    def forward(self, x):
        origin = x
        x = self.ln(x)
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.ln1(x)
        x = self.dropout(x)
        x = self.output_projection(x) + origin
        return x


class Feature_seq_trans(nn.Module):
    def __init__(self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob,
                 n_layers, layer_norm_eps=1e-12):
        super(Feature_seq_trans, self).__init__()
        self.n_layers = n_layers
        self.multi_head_attention_list = nn.ModuleList(DIY_mutlihead_attn(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
                                                       for _ in range(self.n_layers))
        self.mlp_list = nn.ModuleList(MLP_ln_trans(input_dim=hidden_size, embed_dim=intermediate_size,
                                                   output_dim=hidden_size, dropout=hidden_dropout_prob)
                                      for _ in range(self.n_layers))

    def forward(self, query_tensor, key_tensor, x, attention_mask):
        feature_list = []
        for layer in range(self.n_layers):
            x = self.multi_head_attention_list[layer](query_tensor, key_tensor, x, attention_mask)
            x = self.mlp_list[layer](x)
            feature_list.append(x)
        x = sum(feature_list) / len(feature_list)
        return x


class DIY_mutlihead_attn(nn.Module):
    def __init__(
            self,
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
    ):
        super(DIY_mutlihead_attn, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.pre_ln_q = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.pre_ln_k = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.pre_ln_v = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, query_tensor, key_tensor, value_tensor, attention_mask):
        query_tensor = self.pre_ln_q(query_tensor)
        key_tensor = self.pre_ln_k(key_tensor)
        value_tensor = self.pre_ln_v(value_tensor)
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.post_ln(context_layer)
        context_layer = self.out_dropout(context_layer)
        hidden_states = self.dense(context_layer) + value_tensor
        return hidden_states


class gnn_affinity_soft_attention(nn.Module):
    def __init__(self, emb_dim):
        super(gnn_affinity_soft_attention, self).__init__()
        self.emb_dim = emb_dim
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_alpha = nn.Linear(self.emb_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.long_middle_short = fusion_triple_feature(self.emb_dim)

    def forward(self, mask, short_feature, seq_feature, affinity_feature):
        q1 = self.linear1(short_feature).view(short_feature.size(0), 1, short_feature.size(1))
        q2 = self.linear2(seq_feature)
        q3 = self.linear3(affinity_feature).view(short_feature.size(0), 1, short_feature.size(1))
        alpha = self.sigmoid(q1 + q2 + q3)
        alpha = self.linear_alpha(alpha)
        long_feature = torch.sum(alpha * seq_feature * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.long_middle_short(long_feature, short_feature, affinity_feature)
        return seq_output


class fusion_triple_feature(nn.Module):
    def __init__(self, emb_dim):
        super(fusion_triple_feature, self).__init__()
        self.emb_dim = emb_dim
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.softmax = nn.Softmax(dim=1)
        self.linear_final = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, seq_hidden, pos_emb, category_seq_hidden):
        seq_hidden = seq_hidden.unsqueeze(dim=1)
        pos_emb = pos_emb.unsqueeze(dim=1)
        category_seq_hidden = category_seq_hidden.unsqueeze(dim=1)
        seq_hidden = self.linear1(seq_hidden)
        pos_emb = self.linear2(pos_emb)
        category_seq_hidden = self.linear3(category_seq_hidden)
        fusion_feature = torch.cat((seq_hidden, pos_emb, category_seq_hidden), dim=1)
        attn_weight = self.softmax(fusion_feature)
        fusion_feature = torch.sum(attn_weight * fusion_feature, dim=1)
        fusion_feature = self.linear_final(fusion_feature)
        return fusion_feature


class seq_affinity_soft_attention(nn.Module):
    def __init__(self, emb_dim):
        super(seq_affinity_soft_attention, self).__init__()
        self.emb_dim = emb_dim
        self.linear_1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_alpha = nn.Linear(self.emb_dim, 1, bias=False)
        self.long_middle_short = fusion_triple_feature(self.emb_dim)

    def forward(self, mask, short_feature, seq_feature, affinity_feature):
        q1 = self.linear_1(seq_feature)
        q2 = self.linear_2(short_feature)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q3 = self.linear_3(affinity_feature)
        q3_expand = q3.unsqueeze(1).expand_as(q1)
        alpha = self.linear_alpha(mask * torch.sigmoid(q1 + q2_expand + q3_expand))
        long_feature = torch.sum(alpha.expand_as(seq_feature) * seq_feature, 1)
        seq_output = self.long_middle_short(long_feature, short_feature, affinity_feature)
        return seq_output


class gate_mechanism(nn.Module):
    def __init__(self, emb_dim):
        super(gate_mechanism, self).__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(3 * self.emb_dim, self.emb_dim)
        self.act = nn.Sigmoid()

    def forward(self, tensor_a, tensor_b):
        alpha = torch.cat((tensor_a, tensor_b, tensor_a * tensor_b), dim=-1)
        alpha = self.linear(alpha)
        alpha = self.act(alpha)
        output = alpha * tensor_a + (1 - alpha) * tensor_b
        return output


class Ti_ACG(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(Ti_ACG, self).__init__(config, dataset)
        self.dropout = config['dropout']
        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.max_seq_length = dataset.field2seqlen[self.ITEM_SEQ]
        self.mask_token = self.n_items
        self.gnn_layer_num = config['gnn_layer_num']
        self.gnn_head_num = config['gnn_head_num']
        self.time_layer_num = config['time_layer_num']
        self.time_head_num = config['time_head_num']
        self.loss_weight = config['loss_weight']
        self.temperature_parameter = config['temperature_parameter']
        self.time_span = config['time_span']
        self.timestamp = config['TIME_FIELD'] + '_list'
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.GnnConv_list = nn.ModuleList(TransformerConv(
            in_channels=self.embedding_size,
            out_channels=self.embedding_size,
            heads=self.gnn_head_num,
            beta=True,
            concat=False,
            dropout=self.dropout
        ) for _ in range(self.gnn_layer_num))
        self.linear_list_node = nn.ModuleList(nn.Linear(self.embedding_size, self.embedding_size) for _ in range(self.gnn_layer_num))
        self.linear_list_star = nn.ModuleList(nn.Linear(self.embedding_size, self.embedding_size) for _ in range(self.gnn_layer_num))
        self.absolute_pos_Q_embedding = nn.Embedding(self.max_seq_length, self.embedding_size, padding_idx=0)
        self.absolute_pos_K_embedding = nn.Embedding(self.max_seq_length, self.embedding_size, padding_idx=0)
        self.time_matrix_emb_Q_embedding = nn.Embedding(self.time_span + 1, self.embedding_size, padding_idx=0)
        self.time_matrix_emb_K_embedding = nn.Embedding(self.time_span + 1, self.embedding_size, padding_idx=0)
        self.time_trans = SudokuFormer(
            n_layers=self.time_layer_num,
            n_heads=self.time_head_num,
            hidden_size=self.embedding_size,
            intermediate_size=4 * self.embedding_size,
            hidden_dropout_prob=self.dropout,
            attn_dropout_prob=self.dropout
        )
        self.gnn_soft_attention = gnn_affinity_soft_attention(emb_dim=self.embedding_size)
        self.time_soft_attention = seq_affinity_soft_attention(emb_dim=self.embedding_size)
        self.dropout_layer_list = nn.ModuleList(nn.Dropout(p=self.dropout) for _ in range(2))
        self.gate_list = nn.ModuleList(gate_mechanism(self.embedding_size) for _ in range(self.gnn_layer_num + 1))
        self.ce_loss = nn.CrossEntropyLoss()

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def att_out(self, hidden, star_node, batch):
        star_node_repeat = torch.index_select(star_node, 0, batch)
        sim = (hidden * star_node_repeat).sum(dim=-1)
        sim = softmax(sim, batch)
        att_hidden = sim.unsqueeze(-1) * hidden
        output = global_add_pool(att_hidden, batch)
        return output

    def get_time_matrix(self, time_seq):
        time_matrix_i = time_seq.unsqueeze(-1).expand([-1, self.max_seq_length, self.max_seq_length])
        time_matrix_j = time_seq.unsqueeze(1).expand([-1, self.max_seq_length, self.max_seq_length])
        time_matrix = torch.abs(time_matrix_i - time_matrix_j)
        max_time_matrix = (torch.ones_like(time_matrix) * self.time_span).to(self.device)
        time_matrix = torch.where(time_matrix > self.time_span, max_time_matrix, time_matrix).int()
        return time_matrix

    def forward(self, x, edge_index, alias_inputs, item_seq_len, item_seq, time_matrix, batch):
        gnn_input = self.item_embedding(x)
        gnn_input = self.dropout_layer_list[0](gnn_input)
        star_node = global_mean_pool(gnn_input, batch)
        gnn_feature_list = []
        star_node_list = []
        for gnn_layer in range(self.gnn_layer_num):
            gnn_input = self.GnnConv_list[gnn_layer](gnn_input, edge_index)
            star_node_repeat = torch.index_select(star_node, 0, batch)
            gnn_input = self.linear_list_node[gnn_layer](gnn_input)
            star_node_repeat = self.linear_list_star[gnn_layer](star_node_repeat)
            gnn_input = self.gate_list[gnn_layer](gnn_input, star_node_repeat)
            star_node = self.att_out(gnn_input, star_node, batch)
            gnn_feature_list.append(gnn_input)
            star_node_list.append(star_node)
        gnn_output = sum(gnn_feature_list) / len(gnn_feature_list)
        star_node = sum(star_node_list) / len(star_node_list)
        gnn_seq = gnn_output[alias_inputs]
        time_input = self.item_embedding(item_seq)
        time_input = self.dropout_layer_list[0](time_input)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        absolute_pos_Q = self.absolute_pos_Q_embedding(position_ids)
        absolute_pos_K = self.absolute_pos_K_embedding(position_ids)
        time_matrix_emb_Q = self.time_matrix_emb_Q_embedding(time_matrix)
        time_matrix_emb_K = self.time_matrix_emb_K_embedding(time_matrix)
        time_attention_mask = self.get_attention_mask(item_seq)
        time_seq = self.time_trans(
            time_input,
            time_attention_mask,
            absolute_pos_Q,
            absolute_pos_K,
            time_matrix_emb_Q,
            time_matrix_emb_K
        )
        time_seq_mask = item_seq.gt(0).unsqueeze(2).expand_as(time_seq)
        time_seq_mean = torch.mean(time_seq_mask * time_seq, dim=1)  # 时序
        gnn_seq_mask = alias_inputs.gt(0)
        affinity_feature = self.gate_list[self.gnn_layer_num](time_seq_mean, star_node)
        time_short = self.gather_indexes(time_seq, item_seq_len - 1)
        time_session = self.time_soft_attention(
            mask=time_seq_mask,
            short_feature=time_short,
            seq_feature=time_seq,
            affinity_feature=affinity_feature
        )
        gnn_short = self.gather_indexes(gnn_seq, item_seq_len - 1)
        gnn_session = self.gnn_soft_attention(
            mask=gnn_seq_mask,
            short_feature=gnn_short,
            seq_feature=gnn_seq,
            affinity_feature=affinity_feature
        )
        gnn_session = F.normalize(gnn_session, dim=-1)
        time_session = F.normalize(time_session, dim=-1)
        return time_session, gnn_session

    def calculate_loss(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        time_seq = interaction[self.timestamp]
        time_matrix = self.get_time_matrix(time_seq)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        batch = interaction['batch']
        time_session, gnn_session = self.forward(x, edge_index, alias_inputs, item_seq_len, item_seq, time_matrix, batch)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embedding.weight
        test_item_emb = self.dropout_layer_list[1](test_item_emb)
        test_item_emb = F.normalize(test_item_emb, dim=-1)
        logits = torch.matmul(time_session, test_item_emb.transpose(0, 1)) / self.temperature_parameter
        ce_loss = self.ce_loss(logits, pos_items)
        gnn_logits = torch.matmul(gnn_session, test_item_emb.transpose(0, 1)) / self.temperature_parameter
        gnn_ce_loss = self.ce_loss(gnn_logits, pos_items)
        return self.loss_weight * ce_loss + (1 - self.loss_weight) * gnn_ce_loss

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        time_seq = interaction[self.timestamp]
        time_matrix = self.get_time_matrix(time_seq)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        batch = interaction['batch']
        time_session, gnn_session = self.forward(x, edge_index, alias_inputs, item_seq_len, item_seq, time_matrix, batch)
        test_item_emb = self.item_embedding(test_item)
        test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.mul(time_session + gnn_session, test_item_emb).sum(dim=1) / self.temperature_parameter  # [B]
        return scores

    def full_sort_predict(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        time_seq = interaction[self.timestamp]
        time_matrix = self.get_time_matrix(time_seq)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        batch = interaction['batch']
        time_session, gnn_session = self.forward(x, edge_index, alias_inputs, item_seq_len, item_seq, time_matrix, batch)
        test_items_emb = self.item_embedding.weight
        test_items_emb = F.normalize(test_items_emb, dim=-1)
        scores = torch.matmul(time_session + gnn_session, test_items_emb.transpose(0, 1)) / self.temperature_parameter
        return scores
