
from models.strat_blenderbot_small import Model as strat_blenderbot_small
from models.vanilla_blenderbot_small import Model as vanilla_blenderbot_small
from models.strat_blenderbot_small_no_persona import Model as strat_blenderbot_small_no_persona
# CHANGE: removed DialoGPT imports (strat_dialogpt, vanilla_dialogpt) — unused by PAL

models = {
    'vanilla_blenderbot_small': vanilla_blenderbot_small,
    'strat_blenderbot_small': strat_blenderbot_small,
    'strat_blenderbot_small_no_persona': strat_blenderbot_small_no_persona,
}