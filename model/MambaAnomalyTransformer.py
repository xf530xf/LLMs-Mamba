# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mamba_ssm import Mamba
# from .attn import AnomalyAttention, AttentionLayer
# from .embed import DataEmbedding, TokenEmbedding
# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
# from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, LlamaModel, LlamaConfig
# from peft import LoraConfig, get_peft_model,TaskType
# from peft import LoraConfig, TaskType, get_peft_model
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch

# class CoGatedAttention(nn.Module):
#     def __init__(self, d_model, n_heads, dropout=0.1):
#         super(CoGatedAttention, self).__init__()
#         self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.gate = nn.Linear(d_model * 2, d_model)

#     def forward(self, x, x_1):
#         attn_output, attn_weights = self.attention(x, x_1, x_1)
#         gated_output = torch.sigmoid(self.gate(torch.cat((x, attn_output), dim=-1)))
#         return gated_output, attn_weights


# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, attn_mask=None):
#         new_x, attn, mask, sigma = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )
#         x = x + self.dropout(new_x)
#         y = x = self.norm1(x)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))

#         return self.norm2(x + y), attn, mask, sigma


# class Encoder(nn.Module):
#     def __init__(self, attn_layers, norm_layer=None, d_model=512):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.norm = norm_layer
#         self.mamba = Mamba(
#             d_model=d_model,  # Model dimension d_model
#             d_state=8,  # SSM state expansion factor
#             d_conv=4,  # Local convolution width
#             expand=2,  # Block expansion factor
#         ).to("cuda")
#         self.gate = nn.Linear(d_model * 2, d_model)  # Gate for 0-1 gating

#     def forward(self, x, attn_mask=None):
#         # x [B, L, D]
#         series_list = []
#         prior_list = []
#         sigma_list = []
#         original_x = x  # Save the original input for the skip connection

#         for attn_layer in self.attn_layers:
#             x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
#             x_skip = self.mamba(x) + original_x  # Apply skip connection here
#             x_skip = self.norm(x_skip)
#             gate = torch.sigmoid(self.gate(torch.cat((x, x_skip), dim=-1)))  # 0-1 gating
#             x = gate * x_skip + (1 - gate) * x  # Apply gating

#             series_list.append(series)
#             prior_list.append(prior)
#             sigma_list.append(sigma)
#             original_x = x  # Update original_x for the next layer

#         if self.norm is not None:
#             x = self.norm(x)

#         return x, series_list, prior_list, sigma_list


# class MambaAnomalyTransformer(nn.Module):
#     def __init__(self, win_size, enc_in, c_out, d_model=768, n_heads=8, block_size=10, e_layers=3, d_ff=512,
#                  dropout=0.0, activation='gelu', output_attention=True):
#         super(MambaAnomalyTransformer, self).__init__()
#         self.output_attention = output_attention

#         # Encoding
#         self.embedding = DataEmbedding(enc_in, d_model, dropout)

#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         AnomalyAttention(win_size, d_model, n_heads, block_size, attention_dropout=dropout,
#                                          output_attention=output_attention),
#                         d_model, n_heads),
#                     d_model,
#                     d_ff,
#                     dropout=dropout,
#                     activation=activation
#                 ) for _ in range(e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(d_model),
#             d_model=d_model
#         )

#         self.projection = nn.Linear(d_model, c_out, bias=True)



#     def forward(self, x):
#         enc_out = self.embedding(x)
#         enc_out, series, prior, sigmas = self.encoder(enc_out)
#         enc_out = self.projection(enc_out)


#         if self.output_attention:
#             return enc_out, series, prior, sigmas
#         else:
#             return enc_out  # [B, L, D]



import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, LlamaModel, LlamaConfig
from peft import LoraConfig, get_peft_model,TaskType
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from .cross import CrossModal

###############################################################################

class FeatureFusionMHA(nn.Module):
    def __init__(self, input_dim, num_heads):
        """
        使用多头自注意力融合两个特征向量。

        Args:
            input_dim: 输入特征向量的维度（例如，768）。
            num_heads: 多头注意力的头数。
        """
        super(FeatureFusionMHA, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        # 定义线性变换用于 Q, K, V
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)

        # 输出线性变换
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, feature1, feature2):
        """
        前向传播。

        Args:
            feature1: 第一个特征向量 (torch.Size([batch_size, seq_len, input_dim])).
            feature2: 第二个特征向量 (torch.Size([batch_size, seq_len, input_dim])).

        Returns:
            融合后的特征向量 (torch.Size([batch_size, seq_len, input_dim])).
        """
        batch_size, seq_len, _ = feature1.size()

        # 1. 计算 Q, K, V
        q = self.q_proj(feature1)  # shape: (batch_size, seq_len, input_dim)
        k = self.k_proj(feature2)  # shape: (batch_size, seq_len, input_dim)
        v = self.v_proj(feature2)  # shape: (batch_size, seq_len, input_dim)

        # 2. 拆分多头
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)

        # 3. 计算注意力权重
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # shape: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # shape: (batch_size, num_heads, seq_len, seq_len)

        # 4. 应用注意力
        weighted_values = torch.matmul(attention_weights, v)  # shape: (batch_size, num_heads, seq_len, head_dim)

        # 5. 合并头
        weighted_values = weighted_values.transpose(1, 2).reshape(batch_size, seq_len, self.input_dim)  # shape: (batch_size, seq_len, input_dim)

        # 6. 输出线性变换
        output = self.out_proj(weighted_values)  # shape: (batch_size, seq_len, input_dim)

        return output

import torch
import torch.nn as nn

class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=8, num_encoder_layers=1):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads)
            for _ in range(3)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(3)
        ])
        
        self.word_embedding = word_embedding.T
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine device
        self.to(self.device) # Move the module to the device

    def forward(self, x):
        B = x.shape[0]

        # Move x to the correct device
        x = x.to(self.device)

        # Move word_embedding to the correct device and handle dimensions.  Crucial!
        word_embedding = self.word_embedding.to(self.device)  # copy to keep original embedding intact

        if word_embedding.ndim == 2:
            word_embedding = word_embedding.repeat(B, 1, 1)
        elif word_embedding.shape[0] != B:
            word_embedding = word_embedding[0].repeat(B, 1, 1)


        x = self.linear(x)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        x_time = x

        q = x.transpose(0, 1)

        k = v = word_embedding.transpose(0, 1) # Use moved word_embedding

        for layer, norm in zip(self.cross_attention_layers, self.norms):
            attn_output, _ = layer(q, k, v)
            attn_output = attn_output.transpose(0, 1)
            x = norm(x_time + attn_output)
            x_time = x  # 更新参考点

        return x_time, x

######################################################################################















class CoGatedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CoGatedAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x, x_1):
        attn_output, attn_weights = self.attention(x, x_1, x_1)
        gated_output = torch.sigmoid(self.gate(torch.cat((x, attn_output), dim=-1)))
        return gated_output, attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None, d_model=512):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.mamba = Mamba(
            d_model=d_model,  # Model dimension d_model
            d_state=8,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        ).to("cuda")
        self.gate = nn.Linear(d_model * 2, d_model)  # Gate for 0-1 gating

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        original_x = x  # Save the original input for the skip connection

        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            x_skip = self.mamba(x) + original_x  # Apply skip connection here
            x_skip = self.norm(x_skip)
            gate = torch.sigmoid(self.gate(torch.cat((x, x_skip), dim=-1)))  # 0-1 gating
            x = gate * x_skip + (1 - gate) * x  # Apply gating

            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
            original_x = x  # Update original_x for the next layer

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class MambaAnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=768, n_heads=8, block_size=10, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(MambaAnomalyTransformer, self).__init__()





        
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, d_model, n_heads, block_size, attention_dropout=dropout,
                                         output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            d_model=d_model
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)



        ######################################################
        # MODEL_ID = "TaylorAI/Flash-Llama-30M-48001"
        # self.llama_config = LlamaConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        # self.llama_model = LlamaModel.from_pretrained(
        #     MODEL_ID,
        #     config = self.llama_config
        # )
        
        model_name_or_path = "gpt2"  # 或者使用 "gpt2-medium", "gpt2-large" 等
        self.model = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # changed from CAUSAL_LM
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
                # target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"] #These modules are fine
                bias="none", #Added bias
                modules_to_save = ['wte'] #Added Token Embedding Layer
            )

        self.gpt2 = get_peft_model(self.model, peft_config)


        self.cross=FeatureFusionMHA(768, num_heads=12)

        word_embedding = torch.tensor(torch.load("/data2/NTNU/MAAT/checkpoints/wte_pca_500.pt"))

        self.in_layer = Encoder_PCA(38, word_embedding, hidden_dim=768)    #38  55  25,55

        ########################################################

    def forward(self, x):
        print(x.shape)
        outputs_time1, outputs_text1 = self.in_layer(x) 

        print(outputs_time1.shape)
  
        enc_out = self.embedding(x)

        enc_out1 = self.gpt2(inputs_embeds=enc_out).last_hidden_state 

        # enc_text = self.gpt2(inputs_embeds=outputs_text1).last_hidden_state 

        enc_out1+=outputs_time1

        enc_out2, series, prior, sigmas = self.encoder(enc_out)
        # print("@@@@@@@@@@@",enc_out2.shape,enc_out1.shape)
        enc_fusion=self.cross(enc_out1,enc_out2)   ################
        enc_out = self.projection(enc_fusion)


        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]