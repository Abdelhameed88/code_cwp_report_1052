import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from collections import OrderedDict
from torch.nn import GELU

# Normalization layers dictionary
NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}

def replace_legacy(old_dict):
    """
    Replace legacy keys in a checkpoint dictionary.
    """
    li = []
    for k, v in old_dict.items():
        k = (k.replace('Conv2DwithBN', 'layers')
             .replace('Conv2DwithBN_Tanh', 'layers')
             .replace('Deconv2DwithBN', 'layers')
             .replace('ResizeConv2DwithBN', 'layers'))
        li.append((k, v))
    return OrderedDict(li)

########################
# Building Block Layers
########################

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1,
                 norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels=in_fea, out_channels=out_fea,
                      kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(GELU())
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1,
                 norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [
            nn.Conv2d(in_channels=in_fea, out_channels=out_fea,
                      kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0,
                 output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding)
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(GELU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

########################
# Base and Encoder
########################

class BaseNet(nn.Module):
    """
    Base class for SharedEncoder, DualDecoder, DualDecoder.
    Inherits from nn.Module and defines common attributes.
    """

    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512,
                 sample_spatial=1.0, **kwargs):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.dim4 = dim4
        self.dim5 = dim5
        self.sample_spatial = sample_spatial

class SharedEncoder(BaseNet):
    """
    Encoder class for encoding the inputs.
    Inherits from BaseNet.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.convblock1 = ConvBlock(3, self.dim1, kernel_size=[7, 1], stride=[2, 1], padding=[3, 0])
        self.convblock2_1 = ConvBlock(self.dim1, self.dim2, kernel_size=[3, 1], stride=[2, 1], padding=[1, 0])
        self.convblock2_2 = ConvBlock(self.dim2, self.dim2, kernel_size=[3, 1], padding=[1, 0])
        self.convblock3_1 = ConvBlock(self.dim2, self.dim2, kernel_size=[3, 1], stride=[2, 1], padding=[1, 0])
        self.convblock3_2 = ConvBlock(self.dim2, self.dim2, kernel_size=[3, 1], padding=[1, 0])
        self.convblock4_1 = ConvBlock(self.dim2, self.dim3, kernel_size=[3, 1], stride=[2, 1], padding=[1, 0])
        self.convblock4_2 = ConvBlock(self.dim3, self.dim3, kernel_size=[3, 1], padding=[1, 0])
        self.convblock5_1 = ConvBlock(self.dim3, self.dim3, stride=2)
        self.convblock5_2 = ConvBlock(self.dim3, self.dim3)
        self.convblock6_1 = ConvBlock(self.dim3, self.dim4, stride=2)
        self.convblock6_2 = ConvBlock(self.dim4, self.dim4)
        self.convblock7_1 = ConvBlock(self.dim4, self.dim4, stride=2)
        self.convblock7_2 = ConvBlock(self.dim4, self.dim4)
        self.convblock8 = ConvBlock(self.dim4, self.dim5, kernel_size=[3, ceil(286 * self.sample_spatial / 8)],
                                    padding=0)

    def forward(self, x):
        # x: (batch, 3, H, W)
        x = self.convblock1(x)  
        x = self.convblock2_1(x) 
        x = self.convblock2_2(x) 
        x = self.convblock3_1(x) 
        x = self.convblock3_2(x) 
        x = self.convblock4_1(x) 
        x = self.convblock4_2(x) 
        x = self.convblock5_1(x) 
        x = self.convblock5_2(x) 
        x = self.convblock6_1(x) 
        x = self.convblock6_2(x) 
        x = self.convblock7_1(x)
        x = self.convblock7_2(x) 
        x = self.convblock8(x) 
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x

class DualInputSharedEncoder(BaseNet):
    """
    Encoder class for encoding two distinct inputs.
    Inherits from BaseNet.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.Encoder_Vcomp = SharedEncoder(**kwargs)
        self.Encoder_Hcomp = SharedEncoder(**kwargs)
        # After encoding, merge the two branch outputs with a conv block.
        self.convblock_subsequent = ConvBlock(2 * self.dim5, self.dim5)
    
    def forward(self, x1, x2):
        batch, time, C, H, W = x1.shape
        # Process each time step for the first input:
        x1 = x1.view(batch * time, C, H, W)
        x1 = self.Encoder_Vcomp(x1)  # (batch*time, dim5, 1, 1)
        x1 = x1.view(batch, time, self.dim5, 1, 1)
        
        # Process each time step for the second input:
        x2 = x2.view(batch * time, C, H, W)
        x2 = self.Encoder_Hcomp(x2)  # (batch*time, dim5, 1, 1)
        x2 = x2.view(batch, time, self.dim5, 1, 1)
        
        # Concatenate features along the channel dimension: (batch, time, 2*dim5, 1, 1)
        x = torch.cat((x1, x2), dim=2)
        # Merge batch and time dimensions to apply the subsequent conv block:
        x = x.view(batch * time, 2 * self.dim5, 1, 1)
        x = self.convblock_subsequent(x)  # (batch*time, dim5, 1, 1)
        x = x.view(batch, time, self.dim5, 1, 1)
        return x

########################
# LSTM on the Latent Sequence
########################

class UpdatedNetworkWithLSTM(BaseNet):
    """
    Integrates an LSTM after the DualInputSharedEncoder.
    Processes the full latent sequence from the encoder with an LSTM.
    """
    def __init__(self, lstm_hidden_size=512, lstm_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.encoder = DualInputSharedEncoder(**kwargs)
        # LSTM input size is dim5 from the encoder.
        self.lstm = nn.LSTM(input_size=self.dim5, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        self.lstm_hidden_size = lstm_hidden_size

    def forward(self, x1, x2):
        # x1, x2: (batch, time, 3, H, W)
        # Get encoder outputs: (batch, time, dim5, 1, 1)
        x = self.encoder(x1, x2)
        # Remove spatial dimensions → (batch, time, dim5)
        x = x.squeeze(-1).squeeze(-1)
        # Process the sequence through the LSTM:
        x, _ = self.lstm(x)  # x: (batch, time, lstm_hidden_size)
        batch, time, hidden = x.shape
        # Reshape to (batch*time, hidden, 1, 1) for the decoder.
        x = x.view(batch * time, hidden, 1, 1)
        return x, time

########################
# Decoder(s)
########################

class Decoder(BaseNet):
    """
    Decoder class tailored dynamically for the required output size.
    """
    def __init__(self, in_fea, out_fea, final_output_size, **kwargs):
        super().__init__(**kwargs)
        self.final_output_size = final_output_size

        # Initialize all potential layers
        self.deconv1_1 = DeconvBlock(in_fea, self.dim5, kernel_size=4)
        self.deconv1_2 = ConvBlock(self.dim5, self.dim5)
        self.deconv2_1 = DeconvBlock(self.dim5, self.dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(self.dim4, self.dim4)
        self.deconv3_1 = DeconvBlock(self.dim4, self.dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(self.dim3, self.dim3)
        self.deconv4_1 = DeconvBlock(self.dim3, self.dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(self.dim2, self.dim2)
        self.deconv5_1 = DeconvBlock(self.dim2, self.dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(self.dim1, self.dim1)
        self.deconv6 = ConvBlock_Tanh(self.dim1, out_fea)

        # Predefine transition layers for channel adjustment
        self.channel_adjust = nn.ModuleDict({
            str(self.dim5): nn.Conv2d(self.dim5, self.dim1, kernel_size=1, stride=1),
            str(self.dim4): nn.Conv2d(self.dim4, self.dim1, kernel_size=1, stride=1),
            str(self.dim3): nn.Conv2d(self.dim3, self.dim1, kernel_size=1, stride=1),
            str(self.dim2): nn.Conv2d(self.dim2, self.dim1, kernel_size=1, stride=1),
        })


    def forward(self, x):
        last_channels = self.dim5  # Track the number of channels after the last layer
        if self.final_output_size >= 64:
            x = self.deconv1_1(x)
            x = self.deconv1_2(x)
            last_channels = self.dim5
            x = self.deconv2_1(x)
            x = self.deconv2_2(x)
            last_channels = self.dim4
            x = self.deconv3_1(x)
            x = self.deconv3_2(x)
            last_channels = self.dim3
            x = self.deconv4_1(x)
            x = self.deconv4_2(x)
            last_channels = self.dim2
            x = self.deconv5_1(x)
            x = self.deconv5_2(x)
            last_channels = self.dim1
        elif self.final_output_size == 32:
            x = self.deconv1_1(x)
            x = self.deconv1_2(x)
            last_channels = self.dim5
            x = self.deconv2_1(x)
            x = self.deconv2_2(x)
            last_channels = self.dim4
            x = self.deconv3_1(x)
            x = self.deconv3_2(x)
            last_channels = self.dim3
            x = self.deconv4_1(x)
            x = self.deconv4_2(x)
            last_channels = self.dim2
        elif self.final_output_size == 16:
            x = self.deconv1_1(x)
            x = self.deconv1_2(x)
            last_channels = self.dim5
            x = self.deconv2_1(x)
            x = self.deconv2_2(x)
            last_channels = self.dim4
            x = self.deconv3_1(x)
            x = self.deconv3_2(x)
            last_channels = self.dim3
        elif self.final_output_size == 8:
            x = self.deconv1_1(x)
            x = self.deconv1_2(x)
            last_channels = self.dim5
            x = self.deconv2_1(x)
            x = self.deconv2_2(x)
            last_channels = self.dim4

        # Ensure channel compatibility before `deconv6`
        if last_channels != self.dim1:
            x = self.channel_adjust[str(last_channels)](x)
            
        if x.shape[-2:] != (self.final_output_size, self.final_output_size):
            x = F.interpolate(x, size=(self.final_output_size, self.final_output_size),
                              mode='bilinear', align_corners=True)
    
        # Final output layer
        x = self.deconv6(x)
        return x

########################
# Main Model
########################

class DualEncMultiDcoInvNetWithTailoredDecoders(nn.Module):
    """
    Main model: Takes two input sequences (each of shape [batch, time, 3, H, W]),
    processes them with a shared encoder, fuses them, passes the fused latent sequence
    through an LSTM, and then decodes each time step via two decoders (for different output resolutions).
    """
    def __init__(self, lstm_hidden_size=512, lstm_layers=2, **kwargs):
        super().__init__()
        self.encoder_with_lstm = UpdatedNetworkWithLSTM(lstm_hidden_size=lstm_hidden_size,
                                                        lstm_layers=lstm_layers, **kwargs)
        # Define tailored decoders for each resolution
        self.decoders = nn.ModuleList([
            Decoder(in_fea=self.encoder_with_lstm.lstm_hidden_size, out_fea=1, final_output_size=128, **kwargs),
            Decoder(in_fea=self.encoder_with_lstm.lstm_hidden_size, out_fea=1, final_output_size=256, **kwargs),
        ])

    def forward(self, x1, x2):
        # Ensure inputs have a batch dimension.
        if x1.dim() == 4:  # If shape is (time, C, H, W)
            x1 = x1.unsqueeze(0)  # Now shape becomes (1, time, C, H, W)
        if x2.dim() == 4:
            x2 = x2.unsqueeze(0)

        batch, time, C, H, W = x1.shape  # Now both x1 and x2 are (batch, time, 3, H, W)
        
        # Pass through encoder with LSTM.
        latent, seq_len = self.encoder_with_lstm(x1, x2)
        # latent now has shape: (batch*time, lstm_hidden_size, 1, 1)
        # Recompute batch as (batch*time) // time
        batch = latent.size(0) // seq_len

        outputs = []
        # For each decoder, reshape latent, decode, and then reshape back.
        for decoder in self.decoders:
            # Reshape latent to (batch, time, hidden, 1, 1)
            x_seq = latent.view(batch, seq_len, self.encoder_with_lstm.lstm_hidden_size, 1, 1)
            # Merge batch and time dimensions for the decoder.
            x_seq = x_seq.view(batch * seq_len, self.encoder_with_lstm.lstm_hidden_size, 1, 1)
            decoded = decoder(x_seq)  # (batch*time, out_channels, final_output_size, final_output_size)
            # Reshape back to (batch, time, out_channels, final_output_size, final_output_size)
            decoded = decoded.view(batch, seq_len, decoded.size(1), decoded.size(2), decoded.size(3))
            outputs.append(decoded)
        # Return the outputs from each decoder.
        return outputs[0], outputs[1]

# Optional: Dictionary for model instantiation.
model_dict = {
    'DualEncMultiDcoInvNetWithTailoredDecoders': DualEncMultiDcoInvNetWithTailoredDecoders,
}
