import torch
import clip


def soft_prompt(self):
    token_ids = clip.tokenize("a photo of x x",
                              context_length=self.config.context_length).cuda()

    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=self.config.context_length)
            for tok in self.attributes + self.classes
        ]
    )
    orig_token_embedding = self.clip.token_embedding(tokenized.cuda())

    soft_att_obj = torch.zeros(
        (len(self.attributes)+len(self.classes), orig_token_embedding.size(-1)),
    )

    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_att_obj[id, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    ctx_init = "a photo of "
    n_ctx = len(ctx_init.split())
    prompt = clip.tokenize(
        ctx_init, context_length=self.config.context_length).cuda()
    with torch.no_grad():
        embedding = self.clip.token_embedding(prompt)
    ctx_vectors = embedding[0, 1:1+n_ctx, :]
    return token_ids, soft_att_obj, ctx_vectors
