from libs.model_manager import ModelManager
from structs.agent_struct import AgenticModel


def agent_body():
    mm = ModelManager()
    # mm = None
    agm = AgenticModel(mm)

    return mm, agm
