import os
import sys
import json
import requests

sys.path.insert(0, os.path.join("..", os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

import threading
import logging
import time
from typing import Optional, List

import py_cli_interaction
import uvicorn
import dearpygui.dearpygui as dpg

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.encoders import jsonable_encoder

from common.datamodels import ActionTypeDef
from common.utils import singleton

controller = FastAPI()
logger = logging.getLogger("debug_controller")
logging.basicConfig(level=logging.DEBUG)


class ActionTypeStateMachine:
    __DEFAULT_ACTION_TYPE__ = None
    __ACTION_TYPE_CANDIDATES__ = [
        None,
        ActionTypeDef.to_string(ActionTypeDef.FLING),
        ActionTypeDef.to_string(ActionTypeDef.PICK_AND_PLACE),
        ActionTypeDef.to_string(ActionTypeDef.PICK_AND_PLACE_SINGLE),
        ActionTypeDef.to_string(ActionTypeDef.FOLD_1),
        ActionTypeDef.to_string(ActionTypeDef.FOLD_2),
        ActionTypeDef.to_string(ActionTypeDef.DONE),
    ]

    def __init__(self):
        self._enabled_action_type: Optional[str] = None
        self._hold: bool = False
        pass

    @property
    def action_type(self) -> Optional[str]:
        res = self._enabled_action_type
        return res

    @action_type.setter
    def action_type(self, action_type_str: Optional[str]):
        if action_type_str in self.__ACTION_TYPE_CANDIDATES__:
            self._enabled_action_type = action_type_str
        else:
            pass

    @property
    def action_type_candidates(self) -> List[Optional[str]]:
        return self.__ACTION_TYPE_CANDIDATES__

    def reset(self):
        if not self._hold:
            self._enabled_action_type = self.__DEFAULT_ACTION_TYPE__

    @property
    def hold(self):
        return self._hold

    @hold.setter
    def hold(self, h: bool):
        self._hold = h


state: Optional[ActionTypeStateMachine] = None


def make_response(status_code, **kwargs):
    data = {'code': status_code, 'time+stamp': time.time()}
    data.update(**kwargs)
    json_compatible_data = jsonable_encoder(data)
    resp = JSONResponse(content=json_compatible_data, status_code=status_code)
    return resp


@controller.get("/")
def root():
    return RedirectResponse(url='/docs')


@controller.get("/v1/action_type")
def get_action_type(reset: bool = True):
    global state
    res = state.action_type
    if reset:
        state.reset()
    dpg.set_value("text.state", state.action_type)
    return make_response(status_code=200, action_type=res)


def tui_interaction():
    global state
    while True:
        try:
            sel = py_cli_interaction.must_parse_cli_sel("Select Override Action Type",
                                                        candidates=state.action_type_candidates, default_value=0)
            state.action_type = state.action_type_candidates[sel]
        except KeyboardInterrupt:
            logger.info("got keyboard interrupt, exiting")
            os._exit(1)


def gui_interaction():
    global state, logger

    def set_action_type_btn_callback(sender, app_data, user_data):
        logger.debug(f"sender: {sender} app_data: {app_data} user_data: {user_data}")
        state.action_type = user_data["value"]
        dpg.set_value("text.state", user_data["value"])

    def set_action_type_hold_checkbox_callback(sender, app_data, user_data):
        logger.debug(f"sender: {sender} app_data: {app_data} user_data: {user_data}")
        state.hold = app_data

    dpg.create_context()
    dpg.create_viewport(title='debug_controller', width=300, height=600)

    with dpg.window(label="ActionTypeSelector", width=300, height=400):
        dpg.add_text("Current ActionType")
        with dpg.group():
            text_state = dpg.add_text("None", label="text.state", tag="text.state")

        dpg.add_text("Press buttons to select actions")
        with dpg.group():
            for action in set(state.action_type_candidates):
                dpg.add_button(label=str(action), tag="button." + str(action))  # convert None to 'None'
                dpg.set_item_callback("button." + str(action), set_action_type_btn_callback)
                dpg.set_item_user_data("button." + str(action),
                                       dict(type="action_type", value=action, display=text_state))

        with dpg.group():
            dpg.add_checkbox(label="hold", tag="checkbox.hold")
            dpg.set_item_callback("checkbox.hold", set_action_type_hold_checkbox_callback)
            dpg.set_item_user_data("checkbox.hold", dict(type="action_type"))

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


def entrypoint(argv):
    # Initialize Global Variables
    global state
    state = ActionTypeStateMachine()

    try:
        port = int(argv) if len(argv) > 0 else 8085
    except Exception as e:
        logger.error(e)
        print("Usage: debug_controller.py [port]")
        return

    try:
        logger.info("starting server thread")
        server_thread = threading.Thread(target=uvicorn.run, kwargs=dict(app=controller, port=port, host='0.0.0.0'))
        server_thread.start()

        logger.info("starting interaction thread")
        interaction_thread = threading.Thread(target=gui_interaction)
        interaction_thread.start()

        interaction_thread.join()
        raise KeyboardInterrupt

        # uvicorn.run(app=controller, port=cfg.api_port, host=cfg.api_interface)
    except KeyboardInterrupt:
        print(f"got KeyboardInterrupt, exiting")
        os._exit(1)


@singleton
class Client:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.session = requests.Session()
        assert self.self_test()

    @staticmethod
    def get_instance():
        return Client()

    def self_test(self):
        resp = self.session.get(self.endpoint + "/docs")
        return resp.status_code == 200

    def get_action_type(self, reset: bool = True):
        resp = self.session.get(self.endpoint + "/v1/action_type", params=dict(reset=reset))
        payload = json.loads(resp.content)
        logger.debug(f"payload: {payload}")
        print(payload)
        return ActionTypeDef.from_string(payload["action_type"])


def get_remote_action_type_str(client: Client = None) -> str:
    if client is None:
        logger.debug("client is None, remote ActionType=None")
        return 'null'
    else:
        try:
            res = client.get_action_type()
        except Exception as e:
            logger.warning(e)
            res = None
        logger.debug("remote ActionType={res}")
        return ActionTypeDef.to_string(res)


if __name__ == '__main__':
    import sys

    entrypoint(sys.argv[1:])
