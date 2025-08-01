from libs.model_manager import ModelManager
from agents.lightweight_agent import AgenticModel


def agent_body():
    mm = ModelManager()
    # mm = None
    agm = AgenticModel(mm)

    return mm, agm
