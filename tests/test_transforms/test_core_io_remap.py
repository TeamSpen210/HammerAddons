"""Test the IO remapping behaviour."""
from srctools import Output
from . import Context

from pytest_regressions.file_regression import FileRegressionFixture
from hammeraddons.bsp_transform import apply_io_remaps  # noqa  # private


def test_simple_remap(blank_ctx: Context, file_regression: FileRegressionFixture) -> None:
    """Test some basic remaps."""
    rl_source = blank_ctx.vmf.create_ent('some_ent', targetname='rl_source')
    rl_source.add_out(
        Output('OnMulti', 'some_dest', 'Trigger', delay=0.5),
        Output('OnSingle', 'some_dest', 'Trigger', only_once=True),
        Output('OnTrigger', 'some_dest', 'WillBeRemoved'),
        Output('OnTrigger', 'regular_output', 'SetAnimation', 'dance', 0.125),
        Output('OnSpawn', 'basic_dest', 'NeedsParam', '1'),
        Output('OnSpawn', 'overrides', 'Skin', '5'),
    )

    blank_ctx.add_io_remap(
        'some_dest',
        Output('Trigger', 'model_1', 'Skin', '1'),
        Output('Trigger', 'model_1', 'Skin', '0', delay=1.0, times=3),
    )
    blank_ctx.add_io_remap(
        'some_dest',
        Output('Trigger', 'pfx', 'Start'),
        Output('Trigger', 'pfx', 'Stop', delay=0.5),
    )
    blank_ctx.add_io_remap(
        'overrides',
        Output('Skin', 'another_model', 'Bodygroup', '256'),
    )
    blank_ctx.add_io_remap_removal('some_dest', 'WillBeRemoved')

    apply_io_remaps(blank_ctx)

    file_regression.check(blank_ctx.vmf.export(), encoding='utf8', extension='.vmf')
