from ..models.rome import ROMEHyperParams, apply_rome_to_model
from ..models.memit import MEMITHyperParams, apply_memit_to_model
from ..models.kn import KNHyperParams, apply_kn_to_model
from ..models.mend import MENDHyperParams, MendRewriteExecutor, MendMultimodalRewriteExecutor
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..models.serac import SERACHparams, SeracRewriteExecutor, SeracMultimodalRewriteExecutor
from ..dataset import ZsreDataset, CounterFactDataset, CaptionDataset, VQADataset
from ..models.ike import IKEHyperParams, apply_ike_to_model, apply_ike_to_multimodal_model
from ..models.ft_api import FTApiHyperParams, apply_ft_api_to_model
from ..models.lora import LoRAHyperParams, apply_lora_to_model
from ..models.grace import GraceHyperParams, apply_grace_to_model
from ..models.liveedit import LiveEditMultimodalRewriteExecutor
from ..models.dsca import DSCAMultimodalRewriteExecutor
from ..models.engram import EngramMultimodalRewriteExecutor
from ..models.same_edit.same_edit_main import SAMEEditMultimodalRewriteExecutor
from ..models.time_edit import TIMEEditMultimodalRewriteExecutor

ALG_DICT = {
    'ROME': apply_rome_to_model,
    'MEMIT': apply_memit_to_model,
    "FT": apply_ft_to_model,
    'KN': apply_kn_to_model,
    'MEND': MendRewriteExecutor().apply_to_model,
    'SERAC': SeracRewriteExecutor().apply_to_model,
    'IKE': apply_ike_to_model,
    'FT-Api': apply_ft_api_to_model,
    'LoRA': apply_lora_to_model,
    'GRACE': apply_grace_to_model,
}

ALG_MULTIMODAL_DICT = {
    'MEND': MendMultimodalRewriteExecutor().apply_to_model,
    'ASAM_MEND': MendMultimodalRewriteExecutor().apply_to_model,
    'SERAC': SeracMultimodalRewriteExecutor().apply_to_model,
    'SERAC_MULTI': SeracMultimodalRewriteExecutor().apply_to_model,
    'IKE': apply_ike_to_multimodal_model,
    'LiveEdit': LiveEditMultimodalRewriteExecutor().apply_to_model,
    'LIVEEDIT': LiveEditMultimodalRewriteExecutor().apply_to_model,
    'DSCA': DSCAMultimodalRewriteExecutor().apply_to_model,
    'dsca': DSCAMultimodalRewriteExecutor().apply_to_model,
    'ENGRAM': EngramMultimodalRewriteExecutor().apply_to_model,
    'AI_ENGRAM': EngramMultimodalRewriteExecutor().apply_to_model,
    'AI-ENGRAM': EngramMultimodalRewriteExecutor().apply_to_model,
    'SAME_EDIT': SAMEEditMultimodalRewriteExecutor().apply_to_model,
    'same_edit': SAMEEditMultimodalRewriteExecutor().apply_to_model,
    'SAME-EDIT': SAMEEditMultimodalRewriteExecutor().apply_to_model,
    'TIME': TIMEEditMultimodalRewriteExecutor().apply_to_model,
    'TIME_EDIT': TIMEEditMultimodalRewriteExecutor().apply_to_model,
    'time': TIMEEditMultimodalRewriteExecutor().apply_to_model,
    'time_edit': TIMEEditMultimodalRewriteExecutor().apply_to_model,
}

DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset,
}

MULTIMODAL_DS_DICT = {
    "caption": CaptionDataset,
    "vqa": VQADataset,
}
