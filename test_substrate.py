from substrates.substrate_installer import install_substrate
from absl import app
from absl import flags
from substrates import commons_harvest_uniandes as game
from meltingpot.python.human_players import level_playing_utils

FLAGS = flags.FLAGS

flags.DEFINE_integer('screen_width', 800,
                     'Width, in pixels, of the game screen')
flags.DEFINE_integer('screen_height', 600,
                     'Height, in pixels, of the game screen')
flags.DEFINE_integer('frames_per_second', 8, 'Frames per second of the game')
flags.DEFINE_string('observation', 'RGB', 'Name of the observation to render')
flags.DEFINE_bool('verbose', False, 'Whether we want verbose output')
flags.DEFINE_bool('display_text', False,
                  'Whether we to display a debug text message')
flags.DEFINE_string('text_message', 'This page intentionally left blank',
                    'Text to display if `display_text` is `True`')


_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    'fireZap': level_playing_utils.get_space_key_pressed,
}


def verbose_fn(unused_timestep, unused_player_index: int) -> None:
  pass


def text_display_fn(unused_timestep, unused_player_index: int) -> str:
  return FLAGS.text_message


def main(argv):
  del argv  # Unused.
  install_substrate("commons_harvest_uniandes")
  level_playing_utils.run_episode(
      "WORLD.RGB",
      {},  # Settings overrides
      _ACTION_MAP,
      game.get_config(),
      level_playing_utils.RenderType.PYGAME,
      FLAGS.screen_width, FLAGS.screen_height, FLAGS.frames_per_second,
      verbose_fn if FLAGS.verbose else None,
      text_display_fn if FLAGS.display_text else None)


if __name__ == '__main__':
  app.run(main)
