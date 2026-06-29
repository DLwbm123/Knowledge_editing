from ..trainer import MEND
from ..trainer import SERAC, SERAC_MULTI
from ..trainer import FT
from ..trainer import ASAM_FT, ASAM_MEND
from ..trainer import LiveEdit
from ..trainer import DSCA
from ..trainer.algs.same_edit import SAMEEdit
from ..trainer.algs.time_edit import TIMEEdit


ALG_TRAIN_DICT = {
    'MEND': MEND,
    'ASAM_MEND': ASAM_MEND,
    'SERAC': SERAC,
    'SERAC_MULTI': SERAC_MULTI,
    'FT': FT,
    'ASAM_FT': ASAM_FT,
    'LIVEEDIT': LiveEdit,
    'DSCA': DSCA,
    'SAME_EDIT': SAMEEdit,
    'SAME-EDIT': SAMEEdit,
    'SAMEEDIT': SAMEEdit,
    'TIME': TIMEEdit,
    'TIME_EDIT': TIMEEdit,
}
