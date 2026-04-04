"""Modifies nearby overlays to swap materials."""
from srctools import Vec, conv_float, conv_int, conv_bool
from srctools.bsp import Overlay
from srctools.logger import get_logger
import attrs

from hammeraddons.bsp_transform import trans, Context, ent_description, check_control_enabled


LOGGER = get_logger(__name__)
# Use field instances to verify these are attribute names.
KEYS: list[tuple[str, 'attrs.Attribute[float]']] = [
    ('set_umin', attrs.fields(Overlay).u_min),
    ('set_umax', attrs.fields(Overlay).u_max),
    ('set_vmin', attrs.fields(Overlay).v_min),
    ('set_vmax', attrs.fields(Overlay).v_max),
    ('set_fademindist', attrs.fields(Overlay).fade_min_sq),
    ('set_fademaxdist', attrs.fields(Overlay).fade_max_sq),
]


@trans('comp_overlay_setter')
def overlay_setter(ctx: Context) -> None:
    """Locate nearby overlays, then swaps the materials."""
    for ent in ctx.vmf.by_class['comp_overlay_setter']:
        ent.remove()
        if not check_control_enabled(ent):
            continue
        pos = Vec.from_str(ent['origin'])
        max_dist = conv_float(ent['radius'], 1.0) ** 2
        mat_filter = ent['mat_filter'].casefold()

        desc = ent_description(ent)
        should_remove = conv_bool(ent['remove'])
        new_mat = ent['set_material']
        new_renderorder = conv_int(ent['set_renderorder'], -1)
        if new_renderorder not in range(-1, 4):
            LOGGER.warning(
                '{}: Render order must be between 0-3 or -1, got {}',
                desc, new_renderorder,
            )
            continue
        values = {
            field.name: conv_float(ent[key], None)
            for key, field in KEYS
        }
        if values['fade_min_sq'] is not None:
            values['fade_min_sq'] **= 2
        if values['fade_max_sq'] is not None:
            values['fade_max_sq'] **= 2

        name_filter = ent['name_filter'].casefold()
        valid_ids: set[int] | None = None
        if name_filter:
            valid_ids = {
                conv_int(ent['overlayid'])
                for ent in ctx.vmf.search(name_filter)
                if ent['classname'] == 'info_overlay_accessor'
            }
        found = 0

        for overlay in list(ctx.bsp.overlays):
            if valid_ids is not None and overlay.id not in valid_ids:
                continue
            if mat_filter and overlay.texture.mat.casefold() != mat_filter:
                continue
            if max_dist and (pos - overlay.origin).mag_sq() > max_dist:
                continue
            LOGGER.debug('Overlay setter {}: Found overlay {!r}', desc, overlay)
            found += 1
            if should_remove:
                ctx.bsp.overlays.remove(overlay)
                continue  # No point modifying anything else.
            if new_mat:
                old = overlay.texture
                overlay.texture = ctx.bsp.create_texinfo(new_mat, copy_from=overlay.texture, fsys=ctx.pack.fsys)
            if new_renderorder != -1:
                overlay.render_order = new_renderorder
            for field_name, val in values.items():
                if val is not None:
                    setattr(overlay, field_name, val)
        if not found:
            LOGGER.warning(
                '{}: No overlays found within {} from {}',
                desc, ent['radius'], pos,
            )
        else:
            LOGGER.debug('{}: Found {} overlays', desc, found)
