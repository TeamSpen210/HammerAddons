"""Tweak material_modify_control to avoid the variable showing up in instances."""
from __future__ import annotations
from collections.abc import Iterable, Iterator
from collections import Counter

import srctools.logger
from hammeraddons.bsp_transform import trans, Context
from srctools import Entity, conv_bool
from srctools.bsp import BModel
from srctools.mdl import Model
from srctools.tokenizer import TokenSyntaxError
from srctools.vmt import Material


LOGGER = srctools.logger.get_logger(__name__)


def iter_materials(
    ctx: Context,
    model_mats: dict[str, Iterable[str]],
    ent: Entity,
) -> Iterator[str]:
    """Find materials this entity uses."""
    try:
        bsp_model: BModel = ctx.bsp.bmodels[ent]
    except KeyError:
        # This could be a prop model.
        prop_model = ent['model']
        if prop_model:
            try:
                yield from model_mats[prop_model]
            except KeyError:
                # Get all the materials this model uses.
                try:
                    mdl = Model(ctx.pack.fsys, ctx.pack.fsys[prop_model])
                except FileNotFoundError:
                    LOGGER.warning(
                        'Model "{}" does not exist for "{}"',
                        prop_model, ent['targetname'],
                    )
                    model_mats[prop_model] = ()
                except ValueError:
                    LOGGER.warning('Invalid model "{}"', prop_model)
                    model_mats[prop_model] = ()
                else:
                    ent_materials = model_mats[prop_model] = list(mdl.iter_textures())
                    yield from ent_materials
                del mdl  # Complicated.
    else:  # A BSP model.
        for face in bsp_model.faces:
            if (texinfo := face.texinfo) is not None:
                yield texinfo.mat

    if ent['classname'] == 'info_overlay_accessor':
        yield ent['material']


@trans('FGD - material_modify_control')
def material_modify_control(ctx: Context) -> None:
    """Prepend $ to mat-mod-control variable keyvalues if required.

    This allows Hammer to not detect this as a fixup variable.
    """
    # Material name -> does it have either materialmodify or materialmodifyanimated proxies.
    mat_has_proxy: dict[str, bool] = {}
    # Model or "*xx" index -> list of relevant materials.
    model_mats: dict[str, Iterable[str]] = {}
    fsys = ctx.pack.fsys  # Close over just filesystem.

    def material_has_proxy(mat_name: str) -> bool:
        """Check if this material has the proxy."""
        lowered = mat_name.casefold()
        if not lowered.startswith(('materials/', 'materials\\')):
            mat_name = f'materials/{mat_name}'
            lowered = f'materials/{lowered}'
        if not lowered.endswith('.vmt'):
            mat_name += '.vmt'
            lowered += '.vmt'
        try:
            return mat_has_proxy[lowered]
        except KeyError:
            pass
        try:
            file = fsys[mat_name]
        except FileNotFoundError:
            LOGGER.warning('Material "{}" does not exist!', mat_name)
            mat_has_proxy[lowered] = False
            return False
        with file.open_str() as f:
            try:
                mat = Material.parse(f, mat_name)
                mat = mat.apply_patches(fsys)
            except (TokenSyntaxError, ValueError) as exc:
                LOGGER.warning('Material "{}" is not valid: ', exc_info=exc)
        for proxy in mat.proxies:
            if proxy.name in ('materialmodify', 'materialmodifyanimated'):
                mat_has_proxy[lowered] = True
                return True
        mat_has_proxy[lowered] = False
        return False

    matmod: Entity
    for matmod in ctx.vmf.by_class['material_modify_control']:
        var_name = matmod['materialvar']
        if var_name and not var_name.startswith(('$', '%')):
            matmod['materialvar'] = '$' + var_name

        if not conv_bool(matmod.pop('srctools_search_parent')):
            continue
        filter_mat = matmod['materialname'].casefold()

        targets: set[tuple[str, str]] = set()
        found_count: Counter[str] = Counter()
        for parent in ctx.vmf.search(matmod['parentname']):
            found_count[parent['targetname']] += 1
            for mat in set(iter_materials(ctx, model_mats, parent)):
                if material_has_proxy(mat) and (not filter_mat or filter_mat in mat.casefold()):
                    targets.add((parent['targetname'], mat))
        duplicates = [f'- {name}' for name, count in found_count.items() if count > 1]
        if duplicates:
            LOGGER.warning(
                '"{}" has multiple entities with the same name in parents! '
                'Only one with each name will be affected:\n{}',
                matmod['targetname'],
                '\n'.join(duplicates),
            )
        LOGGER.debug('"{}": {} ents to generate', matmod['targetname'], len(targets))
        if not targets:
            LOGGER.warning('"{}"\'s parent has no valid materials!', matmod['targetname'])
            # Leave it unchanged, in case the material name happens to actually be correct.
        else:
            targ_iter = iter(targets)
            matmod['parentname'], matmod['materialname'] = next(targ_iter)
            for parent_name, extra_mat in targ_iter:
                matmod_extra = matmod.copy()
                ctx.vmf.add_ent(matmod_extra)
                matmod_extra['parentname'] = parent_name
                matmod_extra['materialname'] = extra_mat
