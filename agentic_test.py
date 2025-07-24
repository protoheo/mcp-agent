from libs.model_manager import ModelManager
from structs.agent_struct import AgenticModel


if __name__ == '__main__':
    mm = ModelManager()
    # mm = None
    agm = AgenticModel(mm)

    while True:
        usr_prompt = input("Chat:")
        if usr_prompt == 'ee':
            break

        ret = agm.run_chat(usr_prompt)
        if len(ret) > 4:
            break
