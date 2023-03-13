from multilingual_clip import Config_MCLIP
from typing import Optional, Union
import os
import transformers
import torch


class MultilingualCLIP(transformers.PreTrainedModel):
    config_class = Config_MCLIP.MCLIPConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        # Kwargs, such as cache_dir, and local_files_only are absorbed by the inherited from_pretrained,
        # they need to be forwarded to the next model as well.
        settings_kwargs = copy_settings_kwargs(**kwargs)
        slf = super().from_pretrained(pretrained_model_name_or_path, model_args, **kwargs)
        slf.transformer = transformers.AutoModel.from_pretrained(slf.config.modelBase, **settings_kwargs)
        slf.LinearTransformation = torch.nn.Linear(in_features=slf.config.transformerDimensions,
                                                   out_features=slf.config.numDims)
        return slf

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def forward(self, txt, tokenizer):
        txt_tok = tokenizer(txt, padding=True, return_tensors='pt')
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.LinearTransformation(embs)

    def forward_after_tokenization(self, input_tensor):
        """
        input_tensor: tokenized vector and attention vector
        """
        att = input_tensor['attention_mask']
        embs = self.transformer(**input_tensor)[0]
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.LinearTransformation(embs)

    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict, pretrained_model_name_or_path, _fast_init=True):
        model.load_state_dict(state_dict)
        return model, [], [], []


def copy_settings_kwargs(**kwargs):
    """
    cache_dir (`Union[str, os.PathLike]`, *optional*):
        Path to a directory in which a downloaded pretrained model configuration should be cached if the
        standard cache should not be used.
    force_download (`bool`, *optional*, defaults to `False`):
        Whether or not to force the (re-)download of the model weights and configuration files, overriding the
        cached versions if they exist.
    resume_download (`bool`, *optional*, defaults to `False`):
        Whether or not to delete incompletely received files. Will attempt to resume the download if such a
        file exists.
    proxies (`Dict[str, str]`, *optional*):
        A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
        'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
    output_loading_info(`bool`, *optional*, defaults to `False`):
        Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
    local_files_only(`bool`, *optional*, defaults to `False`):
        Whether or not to only look at local files (i.e., do not try to download the model).
    use_auth_token (`str` or `bool`, *optional*):
        The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
        the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
    """
    subset_kwargs = dict()
    subset_kwargs["cache_dir"] = kwargs.get("cache_dir")
    subset_kwargs["force_download"] = kwargs.get("force_download")
    subset_kwargs["resume_download"] = kwargs.get("resume_download")
    subset_kwargs["proxies"] = kwargs.get("proxies")
    subset_kwargs["output_loading_info"] = kwargs.get("output_loading_info")
    subset_kwargs["local_files_only"] = kwargs.get("local_files_only")
    subset_kwargs["use_auth_token"] = kwargs.get("use_auth_token")
    return subset_kwargs
