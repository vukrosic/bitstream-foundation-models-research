# Byte Latent Transformer (BLT) Research

YouTube video - https://youtu.be/SLtLP6J9xTk

Bilibili video - https://www.bilibili.com/video/BV1BjtRzMER3/

## Research Questions & Future Directions

### Entropy Model


First we need a byte level LLM that will groups bytes into patches based on uncertainty of the next byte.

Train on same dataset (or should it be smaller? research question) to predict the next byte.

Vocab size of entropy model in BLT is 265, just all bytes, no EOS or PAD.