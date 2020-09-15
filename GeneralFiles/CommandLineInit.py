import vizdoom as vzd
from argparse import ArgumentParser
DEFAULT_CONFIG = "../scenarios/my_way_home_allpoints.cfg"
parser = ArgumentParser("ViZDoom example showing how to use information about objects and map.")
parser.add_argument(dest="config",
                    default=DEFAULT_CONFIG,
                    nargs="?",
                    help="Path to the configuration file of the scenario."
                         " Please see "
                         "../scenarios/*cfg for more scenarios.")
args = parser.parse_args()
game = vzd.DoomGame()
game.load_config(args.config)
game.set_render_hud(False)
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_objects_info_enabled(True)
game.set_sectors_info_enabled(True)
game.clear_available_game_variables()
game.add_available_game_variable(vzd.GameVariable.POSITION_X)
game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
game.add_available_game_variable(vzd.GameVariable.ANGLE)
game.init()
game.new_episode()
actions = [[0, True, False, False],
           [90, False, True, False],
           [-90, False, False, True]]